"""Compiler-drafted constrained decoding tests."""

from __future__ import annotations

import json
from collections import Counter

import pytest
import torch

from slm_training.dsl.grammar.fastpath.compiler_draft import (
    CompletionForest,
    CompletionPath,
    ConstraintEvidence,
    ConstraintStage,
    active_declaration_binder_id,
    active_declaration_reference_count,
    active_parent_component_ids,
    binder_reference_arities,
    build_completion_forest,
    gold_compiler_decisions,
    gold_compiler_decision_positions,
    root_declaration_reference_arity_target,
    semantic_component_edges,
)
from slm_training.dsl.schema import ExampleRecord
from slm_training.data.contract import (
    GenerationRequest,
    RuntimeSymbol,
    canonicalize_example_template_markers,
)
from slm_training.models.blocks import DenoiserTower
from slm_training.models.decode_stats import collect_decode_stats
from slm_training.models.dsl_tokenizer import DSLNativeTokenizer
from slm_training.models.grammar import make_grammar_state
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel
from slm_training.harnesses.distill.trace_store import DecodeTraceRecorder
from slm_training.harnesses.preference.local_decisions import events_from_trace
from slm_training.levers import (
    PROHIBITED_TEMPLATE_SEMANTIC_LEVERS,
    SLOT_CONTRACT_DECODE_LEVERS,
)


def _model(**config_overrides) -> TwoTowerModel:
    record = ExampleRecord(
        id="compiler",
        prompt="card",
        openui='root = Card([title])\ntitle = TextContent(":hero.title")\n',
        placeholders=[":hero.title"],
        split="train",
        source="fixture",
    )
    output_tokenizer = config_overrides.pop("output_tokenizer", "lexer")
    compiler_decode_mode = config_overrides.pop("compiler_decode_mode", "tree")
    satisfy_companions = config_overrides.pop("_satisfy_companions", True)
    active_contract_weight = any(
        float(config_overrides.get(name, 0.0) or 0.0) > 0.0
        for name in (*SLOT_CONTRACT_DECODE_LEVERS, "semantic_role_decode_weight")
    )
    if active_contract_weight and satisfy_companions:
        config_overrides.setdefault("slot_contract_in_context", True)
        config_overrides.setdefault("slot_contract_constrained_decode", True)
    if (
        satisfy_companions
        and float(config_overrides.get("semantic_role_decode_weight", 0.0) or 0.0) > 0.0
    ):
        config_overrides.setdefault("honest_slot_contract", True)
        config_overrides.setdefault("semantic_role_contract_in_context", True)
    prohibited_overrides = {
        name: config_overrides.pop(name)
        for name in PROHIBITED_TEMPLATE_SEMANTIC_LEVERS
        if name in config_overrides
    }
    config = TwoTowerConfig(
        context_backend="scratch",
        output_tokenizer=output_tokenizer,
        compiler_decode_mode=compiler_decode_mode,
        d_model=32,
        n_heads=2,
        context_layers=1,
        denoiser_layers=1,
        max_prompt_len=32,
        max_target_len=32,
        grammar_ltr_max_tokens=32,
        gen_steps=1,
        seed=0,
        **config_overrides,
    )
    model = TwoTowerModel.from_records(
        [canonicalize_example_template_markers(record)], config=config, device="cpu"
    )
    # Internal bias tests exercise legacy helpers directly. Production config
    # construction rejects these values; restore only after that boundary.
    for name, value in prohibited_overrides.items():
        setattr(model.config, name, value)
    model.eval()
    return model


def test_gathered_projection_matches_full_lm_head() -> None:
    denoiser = DenoiserTower(vocab_size=17, d_model=8, n_layers=1, n_heads=2)
    hidden = torch.randn(2, 8)
    candidates = torch.tensor([1, 5, 11])
    gathered = denoiser.project(hidden, candidates)
    full = denoiser.project(hidden)
    assert torch.allclose(gathered, full.index_select(-1, candidates))


def test_choice_decode_all_singletons_skips_denoiser(monkeypatch) -> None:
    from slm_training.models import choice_tokenizer

    model = _model(output_tokenizer="choice")

    class SingletonState:
        def __init__(self, tokenizer, *, slot_count=0):
            self.tokenizer = tokenizer

        def allowed_ids(self, _remaining):
            return {self.tokenizer.eos_id}

    monkeypatch.setattr(choice_tokenizer, "ChoiceDecodeState", SingletonState)
    monkeypatch.setattr(
        model,
        "_denoiser_forward",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("all-singleton choice step must not run the denoiser")
        ),
    )

    def forbidden_scoring(*_args, **_kwargs):
        raise AssertionError("exact singleton must commit before semantic scoring")

    for helper in (
        "_component_inventory_bias",
        "_component_plan_bias",
        "_slot_component_bias",
        "_schema_role_slot_bias",
        "_slot_coverage_close_bias",
        "_semantic_plan_bias",
        "_semantic_plan_inline_bias",
        "_semantic_plan_root_bias",
    ):
        monkeypatch.setattr(model, helper, forbidden_scoring)
    ctx, ctx_pad = model._encode_context(["one", "two"])
    with collect_decode_stats() as stats:
        ids = model._choice_ltr_decode_batch(ctx, ctx_pad, 8, [None, None])

    assert ids[:, 1].tolist() == [model.tokenizer.eos_id] * 2
    assert stats.forced_row_tokens_without_forward == 2
    assert stats.all_forced_steps_without_forward == 1
    assert stats.ambiguous_rows_forwarded == 0


def test_choice_decode_mixed_step_forwards_only_ambiguous_rows(monkeypatch) -> None:
    from slm_training.models import choice_tokenizer

    model = _model(output_tokenizer="choice")
    other_id = model.tokenizer.token_to_id["+Card"]

    class MixedState:
        def __init__(self, tokenizer, *, slot_count=0):
            self.tokenizer = tokenizer
            self.slot_count = slot_count
            self.current_marker = None
            self.mode = None
            self.frames = []

        def allowed_ids(self, _remaining):
            if self.slot_count:
                return {self.tokenizer.eos_id, other_id}
            return {self.tokenizer.eos_id}

    forwarded_rows = []

    def forward(ids, _ctx, _ctx_pad):
        forwarded_rows.append(int(ids.size(0)))
        logits = torch.zeros((ids.size(0), ids.size(1), model.tokenizer.vocab_size))
        logits[..., model.tokenizer.eos_id] = 100.0
        return logits

    monkeypatch.setattr(choice_tokenizer, "ChoiceDecodeState", MixedState)
    monkeypatch.setattr(model, "_denoiser_forward", forward)
    ctx, ctx_pad = model._encode_context(["forced", "ambiguous"])
    with collect_decode_stats() as stats:
        ids = model._choice_ltr_decode_batch(ctx, ctx_pad, 8, [None, [":hero.title"]])

    assert forwarded_rows == [1]
    assert ids[:, 1].tolist() == [model.tokenizer.eos_id] * 2
    assert stats.forced_row_tokens_without_forward == 1
    assert stats.all_forced_steps_without_forward == 0
    assert stats.ambiguous_rows_forwarded == 1


@pytest.mark.parametrize(
    ("coverage", "expected_forwards", "expected_forced"),
    [("complete", 0, 1), ("partial", 1, 0)],
)
def test_compiler_singleton_bypass_requires_complete_coverage(
    monkeypatch, coverage, expected_forwards, expected_forced
) -> None:
    from slm_training.dsl.grammar.fastpath import compiler_draft

    model = _model()
    forest = CompletionForest(
        (CompletionPath((model.tokenizer.eos_id,), "eos"),), coverage
    )
    monkeypatch.setattr(
        compiler_draft, "build_completion_forest", lambda *_a, **_k: forest
    )
    original_hidden = model._denoiser_hidden
    forwards = 0

    def hidden(*args, **kwargs):
        nonlocal forwards
        forwards += 1
        return original_hidden(*args, **kwargs)

    monkeypatch.setattr(model, "_denoiser_hidden", hidden)
    ctx, ctx_pad = model._encode_context(["card"])
    with collect_decode_stats() as stats:
        result = model._compiler_ltr_decode_one(
            ctx, ctx_pad, 8, mode="restricted", slot_contract=None
        )

    assert int(result[1]) == model.tokenizer.eos_id
    assert forwards == expected_forwards
    assert stats.forced_tokens == expected_forced


def test_compiler_empty_forest_records_bounded_dead_end_trace(monkeypatch) -> None:
    from slm_training.dsl.grammar.fastpath import compiler_draft

    model = _model()
    forest = CompletionForest((), "complete", terminals=("NAME",))
    monkeypatch.setattr(
        compiler_draft, "build_completion_forest", lambda *_a, **_k: forest
    )
    ctx, ctx_pad = model._encode_context(["card"])

    with collect_decode_stats() as stats:
        model._compiler_ltr_decode_one(ctx, ctx_pad, 8, mode="tree", slot_contract=None)

    assert stats.compiler_fallbacks == 1
    assert stats.constrained_dead_ends == 1
    assert stats.constrained_dead_end_last_position == 1
    assert stats.constrained_dead_end_traces == [
        {
            "phase": "compiler_tree",
            "reason": "empty_completion_forest",
            "position": 1,
            "prefix_text": "",
            "prefix_tokens": ["<bos>"],
            "terminals": ["NAME"],
        }
    ]


def test_choice_gold_decisions_classify_component_roles() -> None:
    from slm_training.models.choice_tokenizer import ChoiceTokenizer

    tokenizer = ChoiceTokenizer.build()
    source = 'root = Card([title])\ntitle = TextContent(":hero.title")'
    decisions = gold_compiler_decisions(
        tokenizer,
        tokenizer.encode(source, placeholders=[":hero.title"]),
        slot_contract=[":hero.title"],
    )
    component_roles = [
        decision.kind for decision in decisions if decision.token_kind == "component"
    ]
    # Structural choice streams emit dependencies first and the root last.
    assert component_roles == ["component_bound", "component_root"]
    assert all(len(decision.candidate_ids) > 1 for decision in decisions)


def test_choice_component_plan_trains_without_surface_compiler() -> None:
    record = canonicalize_example_template_markers(
        ExampleRecord(
            id="choice-plan",
            prompt="card with title",
            openui='root = Card([title])\ntitle = TextContent(":hero.title")',
            placeholders=[":hero.title"],
            split="train",
            source="fixture",
        )
    )
    model = TwoTowerModel.from_records(
        [record],
        config=TwoTowerConfig(
            context_backend="scratch",
            output_tokenizer="choice",
            d_model=32,
            n_heads=2,
            context_layers=1,
            denoiser_layers=1,
            max_prompt_len=32,
            max_target_len=64,
            component_plan_loss_weight=1.0,
            component_plan_decode_weight=1.0,
            seed=0,
        ),
        device="cpu",
    )
    loss = model.training_loss([record])
    loss.backward()
    assert torch.isfinite(loss)
    assert model.component_plan_head is not None
    assert model.component_plan_head.weight.grad is not None
    assert model.component_plan_head.weight.grad.abs().sum() > 0
    assert model.last_training_metrics["component_plan_root_accuracy"] >= 0.0


def test_slot_component_head_trains_on_visible_slot_owners() -> None:
    record = canonicalize_example_template_markers(
        ExampleRecord(
            id="slot-components",
            prompt="email field and submit action",
            openui=(
                "root = Stack([field, submit])\n"
                'field = Input("email", ":form.email")\n'
                'submit = Button(":form.submit")'
            ),
            placeholders=[":form.email", ":form.submit"],
            split="train",
            source="fixture",
        )
    )
    model = TwoTowerModel.from_records(
        [record],
        config=TwoTowerConfig(
            context_backend="scratch",
            output_tokenizer="choice",
            d_model=32,
            n_heads=2,
            context_layers=1,
            denoiser_layers=1,
            max_prompt_len=32,
            max_target_len=64,
            slot_component_loss_weight=1.0,
            slot_component_decode_weight=2.0,
            seed=0,
        ),
        device="cpu",
    )

    loss = model.training_loss([record])
    loss.backward()

    assert torch.isfinite(loss)
    assert model.slot_component_head is not None
    assert model.slot_component_head.weight.grad is not None
    assert model.slot_component_head.weight.grad.abs().sum() > 0
    assert model.last_training_metrics["slot_component_rows"] == 2


def test_slot_component_bias_uses_next_unfilled_visible_slot() -> None:
    from types import MethodType

    model = _model(
        slot_component_loss_weight=1.0,
        slot_component_decode_weight=2.0,
    )
    assert model.slot_component_head is not None
    tokenizer = model.tokenizer
    component_ids = model._component_inventory_token_ids()
    component_index = {
        tokenizer.id_to_token[token_id]: index
        for index, token_id in enumerate(component_ids)
    }

    def logits(self, slots, context, pad_mask, context_rows, next_slots=None):
        rows = torch.zeros(
            (len(slots), len(component_ids)),
            dtype=context.dtype,
            device=context.device,
        )
        for index, slot in enumerate(slots):
            target = "Input" if slot == "content:0" else "Button"
            rows[index, component_index[target]] = 3.0
        return rows

    model._slot_component_logits = MethodType(logits, model)
    ctx, ctx_pad = model._encode_context(["email field and submit action"])
    input_id = tokenizer.token_to_id["Input"]
    button_id = tokenizer.token_to_id["Button"]
    candidates = (input_id, button_id)
    kinds = ("component_bound", "component_bound")

    email_bias = model._slot_component_bias(
        ctx,
        ctx_pad,
        [tokenizer.bos_id],
        candidates,
        kinds,
        [":form.email", ":form.submit"],
    )
    submit_bias = model._slot_component_bias(
        ctx,
        ctx_pad,
        [tokenizer.bos_id, tokenizer.sym_id(0)],
        candidates,
        kinds,
        [":form.email", ":form.submit"],
    )

    assert email_bias is not None and email_bias[0] > email_bias[1]
    assert submit_bias is not None and submit_bias[1] > submit_bias[0]


def test_visible_semantic_role_bias_does_not_require_learned_slot_head() -> None:
    model = _model(semantic_role_decode_weight=4.0)
    assert model.slot_component_head is None
    tokenizer = model.tokenizer
    input_id = tokenizer.token_to_id["Input"]
    button_id = tokenizer.token_to_id["Button"]
    candidates = (input_id, button_id)
    kinds = ("component_bound", "component_bound")
    roles = {
        ":form.email": ("Input",),
        ":form.submit": ("Button",),
    }
    ctx, ctx_pad = model._encode_context(["email field and submit action"])

    email_bias = model._slot_component_bias(
        ctx,
        ctx_pad,
        [tokenizer.bos_id],
        candidates,
        kinds,
        [":form.email", ":form.submit"],
        roles,
    )
    submit_bias = model._slot_component_bias(
        ctx,
        ctx_pad,
        [tokenizer.bos_id, tokenizer.sym_id(0)],
        candidates,
        kinds,
        [":form.email", ":form.submit"],
        roles,
    )

    assert email_bias is not None and email_bias[0] == 4.0
    assert email_bias[1] == 0.0
    assert submit_bias is not None and submit_bias[0] == 0.0
    assert submit_bias[1] == 4.0


def test_visible_semantic_roles_gate_unmatched_learned_slot_bias() -> None:
    from types import MethodType

    model = _model(
        slot_component_loss_weight=1.0,
        slot_component_decode_weight=2.0,
        semantic_role_decode_weight=4.0,
    )
    tokenizer = model.tokenizer
    component_ids = model._component_inventory_token_ids()
    component_index = {
        tokenizer.id_to_token[token_id]: index
        for index, token_id in enumerate(component_ids)
    }

    def logits(self, slots, context, pad_mask, context_rows, next_slots=None):
        rows = torch.zeros(
            (len(slots), len(component_ids)),
            dtype=context.dtype,
            device=context.device,
        )
        rows[:, component_index["Button"]] = 3.0
        return rows

    model._slot_component_logits = MethodType(logits, model)
    ctx, ctx_pad = model._encode_context(["email input and submit button"])
    candidates = (
        tokenizer.token_to_id["Input"],
        tokenizer.token_to_id["Button"],
    )
    bias = model._slot_component_bias(
        ctx,
        ctx_pad,
        [tokenizer.bos_id],
        candidates,
        ("component_bound", "component_bound"),
        [":form.email"],
        {":form.email": ("Input",)},
    )

    assert bias is not None
    assert bias.tolist() == [4.0, 0.0]


def test_visible_semantic_roles_abstain_with_incomplete_original_coverage() -> None:
    from types import MethodType

    model = _model(
        slot_component_loss_weight=1.0,
        slot_component_decode_weight=2.0,
        semantic_role_decode_weight=4.0,
    )
    tokenizer = model.tokenizer
    component_ids = model._component_inventory_token_ids()
    button_index = component_ids.index(tokenizer.token_to_id["Button"])

    def logits(self, slots, context, pad_mask, context_rows, next_slots=None):
        rows = torch.zeros(
            (len(slots), len(component_ids)),
            dtype=context.dtype,
            device=context.device,
        )
        rows[:, button_index] = 3.0
        return rows

    model._slot_component_logits = MethodType(logits, model)
    ctx, ctx_pad = model._encode_context(["title modal with body"])
    bias = model._slot_component_bias(
        ctx,
        ctx_pad,
        [tokenizer.bos_id, tokenizer.sym_id(1)],
        (tokenizer.token_to_id["Modal"], tokenizer.token_to_id["Button"]),
        ("component_bound", "component_bound"),
        [":modal.title", ":modal.body"],
        {":modal.title": ("Modal",), ":modal.body": ()},
    )

    assert bias is not None
    assert bias.tolist() == [4.0, 6.0]


def test_slot_coverage_close_bias_closes_legal_frames_after_coverage() -> None:
    from types import SimpleNamespace

    model = _model(output_tokenizer="choice", slot_coverage_close_decode_weight=4.0)
    tokenizer = model.tokenizer
    close_id = tokenizer.token_to_id["]"]
    button_id = tokenizer.token_to_id["+Button"]
    candidates = (button_id, close_id)
    scores = torch.zeros(2)
    typed = SimpleNamespace(
        frames=[
            SimpleNamespace(
                kind="variadic",
                expr_type="array",
                schemas=({"type": "object"},),
                close="]",
            )
        ]
    )
    structural = SimpleNamespace(
        frames=[
            SimpleNamespace(
                kind="variadic",
                expr_type="array",
                schemas=(),
                close="]",
            )
        ]
    )
    slots = [":modal.title", ":modal.body"]
    complete = [tokenizer.bos_id, tokenizer.sym_id(0), tokenizer.sym_id(1)]
    incomplete = [tokenizer.bos_id, tokenizer.sym_id(0)]

    bias = model._slot_coverage_close_bias(typed, complete, candidates, scores, slots)

    assert bias is not None and bias.tolist() == [0.0, 4.0]
    assert (
        model._slot_coverage_close_bias(typed, incomplete, candidates, scores, slots)
        is None
    )
    structural_bias = model._slot_coverage_close_bias(
        structural, complete, candidates, scores, slots
    )
    assert structural_bias is not None and structural_bias.tolist() == [0.0, 4.0]


def test_slot_coverage_close_bias_floors_covered_terminal_stack_close() -> None:
    from types import SimpleNamespace

    model = _model(
        output_tokenizer="choice",
        slot_coverage_close_decode_weight=4.0,
    )
    tokenizer = model.tokenizer
    slot_id = tokenizer.sym_id(0)
    close_id = tokenizer.token_to_id["]"]
    state = SimpleNamespace(
        frames=[
            SimpleNamespace(kind="component", expr_type="element:Stack"),
            SimpleNamespace(
                kind="variadic",
                expr_type="array",
                schemas=(),
                close="]",
            ),
        ]
    )

    bias = model._slot_coverage_close_bias(
        state,
        [tokenizer.bos_id, slot_id],
        (slot_id, close_id),
        torch.tensor([20.0, 1.0]),
        [":body"],
    )

    assert bias is not None
    assert bias.tolist() == [0.0, 23.0]


def test_slot_coverage_close_bias_continues_through_compatible_component() -> None:
    from types import SimpleNamespace

    model = _model(
        output_tokenizer="choice",
        slot_coverage_close_decode_weight=4.0,
    )
    tokenizer = model.tokenizer
    button_id = tokenizer.token_to_id["+Button"]
    close_id = tokenizer.token_to_id["]"]
    state = SimpleNamespace(
        frames=[
            SimpleNamespace(
                kind="component",
                expr_type="element:Stack",
            ),
            SimpleNamespace(
                kind="variadic",
                expr_type="array",
                schemas=(
                    {
                        "anyOf": [
                            {"$ref": "#/$defs/Button"},
                            {"$ref": "#/$defs/TextContent"},
                        ]
                    },
                ),
                close="]",
            ),
        ]
    )

    bias = model._slot_coverage_close_bias(
        state,
        [tokenizer.bos_id],
        (button_id, close_id),
        torch.tensor([0.0, 3.0]),
        [":dialog.confirm"],
        {":dialog.confirm": ("Button",)},
        semantic_role_properties={":dialog.confirm": ("action", "confirm", "label")},
    )

    assert bias is not None
    assert bias.tolist() == [7.0, 0.0]


def test_slot_coverage_close_bias_reaches_slot_through_schema_wrapper() -> None:
    from types import SimpleNamespace

    model = _model(output_tokenizer="choice", slot_coverage_close_decode_weight=4.0)
    tokenizer = model.tokenizer
    buttons_id = tokenizer.token_to_id["+Buttons"]
    close_id = tokenizer.token_to_id["]"]
    state = SimpleNamespace(
        frames=[
            SimpleNamespace(kind="component", expr_type="element:Modal"),
            SimpleNamespace(
                kind="variadic",
                expr_type="array",
                schemas=({"anyOf": [{"$ref": "#/$defs/Buttons"}]},),
                close="]",
            ),
        ]
    )

    bias = model._slot_coverage_close_bias(
        state,
        [tokenizer.bos_id],
        (buttons_id, close_id),
        torch.tensor([0.0, 3.0]),
        [":dialog.confirm"],
        {":dialog.confirm": ("Button",)},
        semantic_role_properties={":dialog.confirm": ("action", "confirm", "label")},
    )

    assert bias is not None
    assert bias.tolist() == [7.0, 0.0]


def test_slot_coverage_wrapper_stops_after_bound_roles_are_covered() -> None:
    from types import SimpleNamespace

    model = _model(output_tokenizer="choice", slot_coverage_close_decode_weight=4.0)
    tokenizer = model.tokenizer
    tab_item_id = tokenizer.token_to_id["+TabItem"]
    close_id = tokenizer.token_to_id["]"]
    tab1_id = tokenizer.sym_id(0)
    state = SimpleNamespace(
        frames=[
            SimpleNamespace(kind="component", expr_type="element:Tabs"),
            SimpleNamespace(
                kind="variadic",
                expr_type="array",
                schemas=({"$ref": "#/$defs/TabItem"},),
                close="]",
            ),
        ]
    )
    slots = [":tabs.tab1", ":tabs.tab2", ":tabs.details.title"]
    candidates = {
        ":tabs.tab1": ("AccordionItem", "TabItem"),
        ":tabs.tab2": ("AccordionItem", "TabItem"),
        ":tabs.details.title": ("Callout", "CardHeader"),
    }
    bindings = {"TabItem": (":tabs.tab1", ":tabs.tab2")}

    second = model._slot_coverage_close_bias(
        state,
        [tokenizer.bos_id, tab1_id],
        (tab_item_id, close_id),
        torch.tensor([0.0, 3.0]),
        slots,
        candidates,
        bindings,
    )
    complete = model._slot_coverage_close_bias(
        state,
        [tokenizer.bos_id, tab1_id, tokenizer.sym_id(1)],
        (tab_item_id, close_id),
        torch.tensor([0.0, 3.0]),
        slots,
        candidates,
        bindings,
    )

    assert second is not None and second.tolist() == [7.0, 0.0]
    assert complete is None


def test_slot_coverage_close_bias_continues_through_compatible_object_property() -> (
    None
):
    from types import SimpleNamespace

    model = _model(
        output_tokenizer="choice",
        slot_coverage_close_decode_weight=4.0,
    )
    tokenizer = model.tokenizer
    details_id = tokenizer.token_to_id["n:details"]
    close_id = tokenizer.token_to_id["}"]
    state = SimpleNamespace(
        frames=[
            SimpleNamespace(
                kind="component",
                expr_type="element:ImageGallery",
            ),
            SimpleNamespace(
                kind="object",
                expr_type="object",
                phase="key",
                property_names=("details",),
                schemas=({"type": "string"},),
                close="}",
            ),
        ]
    )

    bias = model._slot_coverage_close_bias(
        state,
        [tokenizer.bos_id],
        (details_id, close_id),
        torch.tensor([1.0, 5.0]),
        [":gallery.caption"],
        {":gallery.caption": ("ImageGallery",)},
        semantic_role_properties={":gallery.caption": ("caption", "details", "text")},
    )

    assert bias is not None
    assert bias.tolist() == [8.0, 0.0]
    schema_bias = model._slot_coverage_close_bias(
        state,
        [tokenizer.bos_id],
        (details_id, close_id),
        torch.tensor([1.0, 5.0]),
        [":gallery.caption"],
        {":gallery.caption": ("TextContent",)},
        semantic_role_properties={":gallery.caption": ("caption", "details", "text")},
    )
    assert schema_bias is not None and schema_bias.tolist() == [8.0, 0.0]


def test_slot_coverage_close_bias_rejects_wrong_owner_direct_slot() -> None:
    from types import SimpleNamespace

    model = _model(
        output_tokenizer="choice",
        slot_coverage_close_decode_weight=4.0,
    )
    tokenizer = model.tokenizer
    slot_id = tokenizer.sym_id(0)
    close_id = tokenizer.token_to_id["-"]
    state = SimpleNamespace(
        frames=[
            SimpleNamespace(
                kind="component",
                expr_type="element:Button",
                arg_index=0,
                schemas=({"type": "string"},),
                close="-",
            )
        ]
    )

    bias = model._slot_coverage_close_bias(
        state,
        [tokenizer.bos_id],
        (slot_id, close_id),
        torch.tensor([1.0, 5.0]),
        [":auth.email"],
        {":auth.email": ("Input",)},
    )

    assert bias is not None
    assert bias.tolist() == [0.0, 4.0]


def test_slot_coverage_direct_slot_requires_placeholder_property() -> None:
    from types import SimpleNamespace

    model = _model(
        output_tokenizer="choice",
        slot_coverage_close_decode_weight=4.0,
    )
    tokenizer = model.tokenizer
    slot_id = tokenizer.sym_id(0)
    close_id = tokenizer.token_to_id["-"]
    frame = SimpleNamespace(
        kind="component",
        expr_type="element:Input",
        arg_index=0,
        schemas=(
            {"type": "string"},
            {"type": "string", "x-openui-placeholder": True},
            {"type": "string", "enum": ["text", "email"]},
        ),
        close="-",
    )
    state = SimpleNamespace(frames=[frame])
    args = (
        state,
        [tokenizer.bos_id],
        (slot_id, close_id),
        torch.tensor([1.0, 5.0]),
        [":auth.email"],
        {":auth.email": ("Input",)},
    )

    assert model._slot_coverage_close_bias(*args) is None
    frame.arg_index = 1
    bias = model._slot_coverage_close_bias(*args)
    assert bias is not None
    assert bias.tolist() == [8.0, 0.0]
    frame.arg_index = 2
    assert model._slot_coverage_close_bias(*args) is None


def test_slot_coverage_close_bias_escapes_wrong_owner_before_nested_component() -> None:
    from types import SimpleNamespace

    model = _model(
        output_tokenizer="choice",
        slot_coverage_close_decode_weight=2.0,
    )
    tokenizer = model.tokenizer
    switch_id = tokenizer.token_to_id["+SwitchGroup"]
    close_id = tokenizer.token_to_id["-"]
    state = SimpleNamespace(
        frames=[
            SimpleNamespace(
                kind="component",
                expr_type="element:Button",
                arg_index=1,
                schemas=({"type": "string"}, {"type": "object"}),
                close="-",
            )
        ]
    )

    bias = model._slot_coverage_close_bias(
        state,
        [],
        (switch_id, close_id),
        torch.tensor([10.0, 12.0]),
        [":auth.name", ":auth.email"],
        {
            ":auth.name": ("Input", "SwitchGroup"),
            ":auth.email": ("Input", "SwitchGroup"),
        },
    )

    assert bias is not None
    assert bias.tolist() == [0.0, 2.0]


def test_slot_coverage_close_trace_records_owner_and_missing_slots() -> None:
    from types import SimpleNamespace

    from slm_training.models.decode_stats import DecodeStats

    model = _model(
        output_tokenizer="choice",
        slot_coverage_close_decode_weight=2.0,
    )
    tokenizer = model.tokenizer
    button_id = tokenizer.token_to_id["+Button"]
    close_id = tokenizer.token_to_id["]"]
    stats = DecodeStats()
    state = SimpleNamespace(
        frames=[
            SimpleNamespace(
                kind="component",
                expr_type="element:Stack",
                phase="",
                arg_index=0,
            ),
            SimpleNamespace(
                kind="variadic",
                expr_type="array",
                close="]",
                active_property=None,
                phase="",
                arg_index=0,
            ),
        ],
        mode="structural",
    )

    trace = model._record_slot_coverage_close_trace(
        stats,
        row=0,
        position=8,
        state=state,
        prefix=[tokenizer.bos_id, tokenizer.sym_id(0)],
        candidate_ids=(button_id, close_id),
        scores_before=torch.tensor([1.0, 5.0]),
        coverage_bias=torch.tensor([6.0, 0.0]),
        scores_after=torch.tensor([7.0, 5.0]),
        slot_contract=[":dialog.title", ":dialog.confirm"],
    )
    model._finalize_semantic_plan_trace(
        trace,
        candidate_ids=(button_id, close_id),
        scores=torch.tensor([7.0, 8.0]),
    )

    assert trace is not None
    assert trace["phase"] == "slot_coverage_close"
    assert trace["mode"] == "coverage_continue"
    assert trace["missing_slots"] == [":dialog.confirm"]
    assert trace["owner_component"] == "Stack"
    assert trace["chosen_token"] == "+Button"
    assert trace["final_token"] == "]"
    assert trace["changed_after_plan"] is True


def test_slot_coverage_close_trace_labels_owner_escape() -> None:
    from types import SimpleNamespace

    from slm_training.models.decode_stats import DecodeStats

    model = _model(
        output_tokenizer="choice",
        slot_coverage_close_decode_weight=2.0,
    )
    tokenizer = model.tokenizer
    close_id = tokenizer.token_to_id["-"]
    input_id = tokenizer.token_to_id["+Input"]
    stats = DecodeStats()
    state = SimpleNamespace(
        frames=[
            SimpleNamespace(
                kind="component",
                expr_type="element:Button",
                close="-",
                active_property=None,
                phase="",
                arg_index=1,
            )
        ],
        mode="structural",
    )

    trace = model._record_slot_coverage_close_trace(
        stats,
        row=0,
        position=3,
        state=state,
        prefix=[tokenizer.bos_id],
        candidate_ids=(input_id, close_id),
        scores_before=torch.tensor([10.0, 12.0]),
        coverage_bias=torch.tensor([0.0, 2.0]),
        scores_after=torch.tensor([10.0, 14.0]),
        slot_contract=[":auth.email"],
    )

    assert trace is not None
    assert trace["mode"] == "owner_escape"
    assert trace["missing_slots"] == [":auth.email"]
    assert trace["chosen_token"] == "-"


def test_repeated_plan_array_close_bias_targets_nested_repeated_family() -> None:
    from types import SimpleNamespace

    model = _model(
        output_tokenizer="choice",
        semantic_plan_repeated_array_close_margin_decode_weight=2.0,
    )
    tokenizer = model.tokenizer
    card_id = tokenizer.token_to_id["+Card"]
    close_id = tokenizer.token_to_id["]"]
    text_id = tokenizer.token_to_id["+TextContent"]
    candidates = (text_id, close_id)
    model._semantic_plan_action_counts = [{card_id: 2}]
    nested = SimpleNamespace(
        frames=[
            SimpleNamespace(kind="component", expr_type="element:Card"),
            SimpleNamespace(kind="variadic", expr_type="array", item_count=0),
            SimpleNamespace(kind="component", expr_type="element:Stack"),
            SimpleNamespace(
                kind="variadic",
                expr_type="array",
                item_count=1,
                close="]",
            ),
        ]
    )

    bias = model._semantic_plan_repeated_array_close_bias(
        0,
        nested,
        candidates,
        torch.tensor([8.0, 1.0]),
    )

    assert bias is not None
    assert bias.tolist() == [0.0, 9.0]
    direct = SimpleNamespace(
        frames=[
            SimpleNamespace(kind="component", expr_type="element:Card"),
            SimpleNamespace(
                kind="variadic",
                expr_type="array",
                item_count=1,
                close="]",
            ),
        ]
    )
    assert model._semantic_plan_repeated_array_close_bias(
        0, direct, candidates, torch.tensor([8.0, 1.0])
    ).tolist() == [0.0, 9.0]
    model._semantic_plan_action_counts = [{card_id: 1, text_id: 2}]
    repeated_inner = SimpleNamespace(
        frames=[
            SimpleNamespace(kind="component", expr_type="element:Card"),
            SimpleNamespace(kind="variadic", expr_type="array", item_count=0),
            SimpleNamespace(kind="component", expr_type="element:TextContent"),
            SimpleNamespace(
                kind="variadic",
                expr_type="array",
                item_count=1,
                close="]",
            ),
        ]
    )
    assert model._semantic_plan_repeated_array_close_bias(
        0, repeated_inner, candidates, torch.tensor([8.0, 1.0])
    ).tolist() == [0.0, 9.0]
    model._semantic_plan_action_counts = [{card_id: 1}]
    assert (
        model._semantic_plan_repeated_array_close_bias(
            0, nested, candidates, torch.tensor([8.0, 1.0])
        )
        is None
    )


def test_repeated_plan_slot_bias_targets_best_unused_visible_slot() -> None:
    from types import SimpleNamespace

    model = _model(
        output_tokenizer="choice",
        semantic_plan_repeated_slot_margin_decode_weight=2.0,
    )
    tokenizer = model.tokenizer
    card_id = tokenizer.token_to_id["+Card"]
    stack_id = tokenizer.token_to_id["+Stack"]
    text_id = tokenizer.token_to_id["+TextContent"]
    slot0 = tokenizer.sym_id(0)
    slot1 = tokenizer.sym_id(1)
    slot2 = tokenizer.sym_id(2)
    model._semantic_plan_action_counts = [{card_id: 2}]
    model._slot_contracts = [
        [":status.title", ":status.body", ":metric.one", ":metric.two"]
    ]
    state = SimpleNamespace(
        frames=[
            SimpleNamespace(kind="component", expr_type="element:Card"),
            SimpleNamespace(kind="variadic", expr_type="array"),
            SimpleNamespace(kind="component", expr_type="element:Stack"),
            SimpleNamespace(kind="variadic", expr_type="array"),
        ]
    )
    prefix = [tokenizer.bos_id, slot0, slot1, card_id, stack_id]
    candidates = (slot1, slot2, text_id)

    bias = model._semantic_plan_repeated_slot_bias(
        0,
        state,
        prefix,
        candidates,
        torch.tensor([9.0, 2.0, 10.0]),
    )

    assert bias is not None
    assert bias.tolist() == [0.0, 10.0, 0.0]
    assert (
        model._semantic_plan_repeated_slot_bias(
            0,
            state,
            [*prefix, slot2],
            candidates,
            torch.tensor([9.0, 2.0, 10.0]),
        )
        is None
    )


def test_repeated_plan_slot_bias_uses_shared_plan_margin_for_lexer_ownership() -> None:
    from slm_training.models.grammar import make_grammar_state

    model = _model(
        semantic_plan_margin_decode_weight=2.0,
        semantic_plan_repeated_slot_margin_decode_weight=0.0,
    )
    tokenizer = model.tokenizer
    card_id = tokenizer.token_to_id["Card"]
    model._semantic_plan_action_counts = [{card_id: 3}]
    model._slot_contracts = [
        [
            ":toolbar.text",
            ":card1.title",
            ":card1.body",
            ":card2.title",
            ":card2.body",
            ":card3.title",
            ":card3.body",
        ]
    ]
    prefix = [
        tokenizer.bos_id,
        *tokenizer.encode("root = Stack([Card([TextContent(", add_special=False),
    ]
    state = make_grammar_state()
    for token_id in prefix[1:]:
        state.advance_token(tokenizer, token_id)
    toolbar = tokenizer.sym_id(0)
    card1_title = tokenizer.sym_id(1)

    bias = model._semantic_plan_repeated_slot_bias(
        0,
        state,
        prefix,
        (toolbar, card1_title),
        torch.tensor([8.0, 1.0]),
    )

    assert bias is not None
    assert (torch.tensor([8.0, 1.0]) + bias).argmax().item() == 1


def test_repeated_plan_slot_bias_closes_completed_lexer_owner_group() -> None:
    from slm_training.models.grammar import make_grammar_state

    model = _model(
        semantic_plan_margin_decode_weight=2.0,
        semantic_plan_repeated_slot_margin_decode_weight=0.0,
    )
    tokenizer = model.tokenizer
    card_id = tokenizer.token_to_id["Card"]
    model._semantic_plan_action_counts = [{card_id: 3}]
    model._slot_contracts = [
        [
            ":toolbar.text",
            ":card1.title",
            ":card1.body",
            ":card2.title",
            ":card2.body",
            ":card3.title",
            ":card3.body",
        ]
    ]
    prefix = [
        tokenizer.bos_id,
        *tokenizer.encode("root = Stack([Card([TextContent(", add_special=False),
        tokenizer.sym_id(1),
        *tokenizer.encode("), TextContent(", add_special=False),
        tokenizer.sym_id(2),
        *tokenizer.encode(")", add_special=False),
    ]
    state = make_grammar_state()
    for token_id in prefix[1:]:
        state.advance_token(tokenizer, token_id)
    comma = tokenizer.token_to_id[","]
    close = tokenizer.token_to_id["]"]

    bias = model._semantic_plan_repeated_slot_bias(
        0,
        state,
        prefix,
        (comma, close),
        torch.tensor([8.0, 1.0]),
    )

    assert bias is not None
    assert (torch.tensor([8.0, 1.0]) + bias).argmax().item() == 1


def test_repeated_plan_slot_bias_does_not_close_parent_sibling_array() -> None:
    from slm_training.models.grammar import make_grammar_state

    model = _model(semantic_plan_margin_decode_weight=2.0)
    tokenizer = model.tokenizer
    card_id = tokenizer.token_to_id["Card"]
    model._semantic_plan_action_counts = [{card_id: 3}]
    model._slot_contracts = [
        [
            ":card1.title",
            ":card1.body",
            ":card2.title",
            ":card2.body",
            ":card3.title",
            ":card3.body",
        ]
    ]
    prefix = [
        tokenizer.bos_id,
        *tokenizer.encode("root = Stack([Card([TextContent(", add_special=False),
        tokenizer.sym_id(0),
        *tokenizer.encode("), TextContent(", add_special=False),
        tokenizer.sym_id(1),
        *tokenizer.encode(")])", add_special=False),
    ]
    state = make_grammar_state()
    for token_id in prefix[1:]:
        state.advance_token(tokenizer, token_id)

    assert (
        model._semantic_plan_repeated_slot_bias(
            0,
            state,
            prefix,
            (tokenizer.token_to_id[","], tokenizer.token_to_id["]"]),
            torch.tensor([1.0, 8.0]),
        )
        is None
    )


def test_typed_array_nonempty_bias_starts_slot_bearing_authored_array() -> None:
    from types import SimpleNamespace

    model = _model(
        output_tokenizer="choice",
        semantic_plan_typed_array_nonempty_margin_decode_weight=2.0,
    )
    tokenizer = model.tokenizer
    gallery_id = tokenizer.token_to_id["+ImageGallery"]
    object_id = tokenizer.token_to_id["{"]
    close_id = tokenizer.token_to_id["]"]
    model._semantic_plan_action_counts = [{gallery_id: 1}]
    model._slot_contracts = [[":gallery.image", ":gallery.caption"]]
    item_schema = {
        "type": "object",
        "properties": {"src": {"type": "string"}},
        "required": ["src"],
    }
    empty = SimpleNamespace(
        frames=[
            SimpleNamespace(kind="component", expr_type="element:ImageGallery"),
            SimpleNamespace(
                kind="variadic",
                expr_type="array",
                item_count=0,
                schemas=(item_schema,),
                close="]",
            ),
        ]
    )
    candidates = (object_id, close_id)

    bias = model._semantic_plan_typed_array_nonempty_bias(
        0,
        empty,
        [tokenizer.bos_id, gallery_id],
        candidates,
        torch.tensor([1.0, 8.0]),
    )

    assert bias is not None
    assert bias.tolist() == [9.0, 0.0]
    empty.frames[-1].item_count = 1
    assert (
        model._semantic_plan_typed_array_nonempty_bias(
            0,
            empty,
            [tokenizer.bos_id, gallery_id],
            candidates,
            torch.tensor([1.0, 8.0]),
        )
        is None
    )
    empty.frames[-1].item_count = 0
    empty.frames[-1].schemas = ({"type": "number"},)
    assert (
        model._semantic_plan_typed_array_nonempty_bias(
            0,
            empty,
            [tokenizer.bos_id, gallery_id],
            candidates,
            torch.tensor([1.0, 8.0]),
        )
        is None
    )


def test_typed_array_nonempty_bias_can_target_schema_item_start() -> None:
    from types import SimpleNamespace

    model = _model(
        output_tokenizer="choice",
        semantic_plan_typed_array_nonempty_margin_decode_weight=2.0,
        semantic_plan_typed_array_item_margin_decode_weight=2.0,
    )
    tokenizer = model.tokenizer
    gallery_id = tokenizer.token_to_id["+ImageGallery"]
    object_id = tokenizer.token_to_id["{"]
    state_id = tokenizer.token_to_id["$@45"]
    close_id = tokenizer.token_to_id["]"]
    model._semantic_plan_action_counts = [{gallery_id: 1}]
    model._slot_contracts = [[":gallery.image"]]
    state = SimpleNamespace(
        frames=[
            SimpleNamespace(kind="component", expr_type="element:ImageGallery"),
            SimpleNamespace(
                kind="variadic",
                expr_type="array",
                item_count=0,
                schemas=(
                    {
                        "type": "object",
                        "properties": {"src": {"type": "string"}},
                    },
                ),
                close="]",
            ),
        ],
        _minimal_schema_id=lambda _schema: object_id,
    )

    bias = model._semantic_plan_typed_array_nonempty_bias(
        0,
        state,
        [tokenizer.bos_id, gallery_id],
        (object_id, state_id, close_id),
        torch.tensor([1.0, 9.0, 8.0]),
    )

    assert bias is not None
    assert bias.tolist() == [10.0, 0.0, 0.0]


def test_typed_array_item_bias_prefers_schema_allowed_role_wrapper() -> None:
    from types import SimpleNamespace

    model = _model(
        output_tokenizer="choice",
        semantic_plan_typed_array_item_margin_decode_weight=2.0,
    )
    tokenizer = model.tokenizer
    bar_id = tokenizer.token_to_id["+BarChart"]
    text_id = tokenizer.token_to_id["+TextContent"]
    slot_id = tokenizer.sym_id(0)
    close_id = tokenizer.token_to_id["]"]
    model._semantic_plan_action_counts = [{}]
    model._slot_contracts = [[":metric"]]
    model._semantic_role_candidates = [{":metric": ("BarChart",)}]
    state = SimpleNamespace(
        frames=[
            SimpleNamespace(kind="component", expr_type="element:Carousel"),
            SimpleNamespace(kind="variadic", expr_type="array"),
            SimpleNamespace(
                kind="variadic",
                expr_type="array",
                item_count=0,
                schemas=(
                    {
                        "anyOf": [
                            {"$ref": "#/$defs/TextContent"},
                            {
                                "$ref": "#/$defs/BarChart",
                                "type": "object",
                                "properties": {"value": {"x-openui-placeholder": True}},
                            },
                        ]
                    },
                ),
                close="]",
            ),
        ]
    )

    bias = model._semantic_plan_typed_array_nonempty_bias(
        0,
        state,
        [tokenizer.bos_id],
        (bar_id, text_id, slot_id, close_id),
        torch.tensor([1.0, 6.0, 9.0, 8.0]),
    )

    assert bias is not None
    assert bias.tolist() == [10.0, 0.0, 0.0, 0.0]


def test_nested_array_frame_preserves_inner_component_schema() -> None:
    from slm_training.models.choice_tokenizer import ChoiceDecodeState

    model = _model(output_tokenizer="choice")
    tokenizer = model.tokenizer
    state = ChoiceDecodeState(tokenizer, slot_count=1)

    assert state.advance_id(tokenizer.token_to_id["+Carousel"])
    assert state.advance_id(tokenizer.token_to_id["["])
    assert state.advance_id(tokenizer.token_to_id["["])
    assert not state.advance_id(tokenizer.sym_id(0))
    assert state.advance_id(tokenizer.token_to_id["+TextContent"])


def test_typed_array_item_bias_accepts_model_introduced_schema_owner() -> None:
    from types import SimpleNamespace

    model = _model(
        output_tokenizer="choice",
        semantic_plan_typed_array_item_margin_decode_weight=2.0,
    )
    tokenizer = model.tokenizer
    text_id = tokenizer.token_to_id["+TextContent"]
    slot_id = tokenizer.sym_id(0)
    close_id = tokenizer.token_to_id["]"]
    model._semantic_plan_action_counts = [{}]
    model._slot_contracts = [[":metric"]]
    model._semantic_role_candidates = [{":metric": ()}]
    state = SimpleNamespace(
        frames=[
            SimpleNamespace(kind="component", expr_type="element:Carousel"),
            SimpleNamespace(kind="variadic", expr_type="array"),
            SimpleNamespace(
                kind="variadic",
                expr_type="array",
                item_count=0,
                schemas=(
                    {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                    },
                ),
                close="]",
            ),
        ],
        _minimal_schema_id=lambda _schema: text_id,
    )

    bias = model._semantic_plan_typed_array_nonempty_bias(
        0,
        state,
        [tokenizer.bos_id],
        (text_id, slot_id, close_id),
        torch.tensor([1.0, 9.0, 8.0]),
    )

    assert bias is not None
    assert bias.tolist() == [10.0, 0.0, 0.0]


def test_schema_slot_reachability_resolves_public_component_refs() -> None:
    from slm_training.dsl.lang_core import library_schema

    carousel = library_schema()["$defs"]["Carousel"]
    item_schema = carousel["properties"]["children"]["items"]["items"]

    assert TwoTowerModel._schema_can_reach_visible_slot(item_schema)


def test_schema_open_bias_prefers_true_for_authored_visible_component() -> None:
    from types import SimpleNamespace

    model = _model(output_tokenizer="choice", schema_open_decode_weight=2.0)
    tokenizer = model.tokenizer
    modal_id = tokenizer.token_to_id["+Modal"]
    true_id = tokenizer.token_to_id["#true"]
    false_id = tokenizer.token_to_id["#false"]
    model._semantic_plan_action_counts = [{modal_id: 1}]
    state = SimpleNamespace(
        frames=[
            SimpleNamespace(
                kind="component",
                expr_type="element:Modal",
                arg_index=1,
                schemas=({"type": "string"}, {"type": "boolean"}),
                property_names=("title", "open"),
            )
        ]
    )

    bias = model._schema_open_bias(
        0,
        state,
        (true_id, false_id),
        torch.tensor([1.0, 6.0]),
    )

    assert bias is not None
    assert bias.tolist() == [7.0, 0.0]

    model._semantic_plan_action_counts = [{}]
    assert (
        model._schema_open_bias(
            0,
            state,
            (true_id, false_id),
            torch.tensor([1.0, 6.0]),
        )
        is None
    )


@pytest.mark.parametrize(
    "weight_name",
    [
        "schema_role_slot_decode_weight",
        "slot_coverage_close_decode_weight",
        "semantic_plan_typed_array_nonempty_margin_decode_weight",
        "semantic_plan_typed_array_item_margin_decode_weight",
        "semantic_plan_repeated_slot_margin_decode_weight",
    ],
)
def test_contract_gated_decode_weight_without_slot_contract_decode_raises(
    weight_name: str,
) -> None:
    """E617: these biases read ``self._slot_contracts[row]``, which only gets

    populated by ``_generate_batch_once`` when ``slot_contract_constrained_decode``
    (or ``template_fill_decode``) is enabled. E611-E616 replayed a matched
    control/treatment eval with one of these weights set >0 but neither flag
    enabled, so the bias silently no-opped on every decode step regardless of
    weight or checkpoint quality. Fail loud instead of reproducing that footgun.
    """
    expected = (
        "template markers are opaque"
        if weight_name in PROHIBITED_TEMPLATE_SEMANTIC_LEVERS
        else "requires one companion configuration"
    )
    with pytest.raises(ValueError, match=expected):
        if weight_name in PROHIBITED_TEMPLATE_SEMANTIC_LEVERS:
            TwoTowerConfig(**{weight_name: 8.0})
            return
        _model(
            output_tokenizer="choice",
            _satisfy_companions=False,
            **{weight_name: 8.0},
        )


@pytest.mark.parametrize(
    "enabling_flag",
    ["slot_contract_constrained_decode", "template_fill_decode"],
)
def test_semantic_role_weight_remains_prohibited_with_contract_decode(
    enabling_flag: str,
) -> None:
    with pytest.raises(ValueError, match="template markers are opaque"):
        TwoTowerConfig(
            output_tokenizer="choice",
            schema_role_slot_decode_weight=8.0,
            **{enabling_flag: True},
        )


def test_marker_name_cannot_activate_semantic_role_scoring() -> None:
    with pytest.raises(ValueError, match="template markers are opaque"):
        RuntimeSymbol(
            surface=":gallery.image",
            role="external_entity",
            semantic_role="image",
        )


def test_schema_value_bias_penalizes_slots_only_for_enum_arguments() -> None:
    from slm_training.dsl.production_codec import CLOSE, LIT_PREFIX, OPEN_PREFIX
    from slm_training.models.choice_tokenizer import ChoiceDecodeState

    model = _model(output_tokenizer="choice", schema_value_decode_weight=4.0)
    tokenizer = model.tokenizer
    state = ChoiceDecodeState(tokenizer, slot_count=2)
    assert state.advance_id(tokenizer.token_to_id[f"{OPEN_PREFIX}Button"])
    slot_id = tokenizer.sym_id(1)
    close_id = tokenizer.token_to_id[CLOSE]
    scores = torch.zeros(2)

    assert model._schema_value_bias(state, (slot_id, close_id), scores) is None
    assert state.advance_id(tokenizer.sym_id(0))
    assert model._schema_value_bias(state, (slot_id, close_id), scores) is None
    assert state.advance_id(tokenizer.token_to_id[f"{LIT_PREFIX}null"])
    bias = model._schema_value_bias(state, (slot_id, close_id), scores)

    assert bias is not None
    assert bias.tolist() == [-4.0, 0.0]


def test_schema_value_bias_floors_enum_slot_below_best_non_slot() -> None:
    from slm_training.dsl.production_codec import CLOSE, LIT_PREFIX, OPEN_PREFIX
    from slm_training.models.choice_tokenizer import ChoiceDecodeState

    model = _model(output_tokenizer="choice", schema_value_decode_weight=4.0)
    tokenizer = model.tokenizer
    state = ChoiceDecodeState(tokenizer, slot_count=2)
    for token_id in (
        tokenizer.token_to_id[f"{OPEN_PREFIX}Button"],
        tokenizer.sym_id(0),
        tokenizer.token_to_id[f"{LIT_PREFIX}null"],
    ):
        assert state.advance_id(token_id)
    slot_id = tokenizer.sym_id(1)
    close_id = tokenizer.token_to_id[CLOSE]

    bias = model._schema_value_bias(
        state,
        (slot_id, close_id),
        torch.tensor([20.0, 1.0]),
    )

    assert bias is not None
    assert bias.tolist() == [-23.0, 0.0]


def test_choice_tokenizer_rejects_unknown_enum_literal() -> None:
    from slm_training.dsl.language_contract import OutputContractError

    model = _model(output_tokenizer="choice")
    tokenizer = model.tokenizer
    source = 'root = Callout("invalid", ":title", ":body")'

    with pytest.raises(OutputContractError, match="forbids free-form strings"):
        tokenizer.encode(source, placeholders=[":title", ":body"])


def test_schema_enum_finalize_replaces_only_invalid_fixed_literal() -> None:
    model = _model(output_tokenizer="choice")
    tokenizer = model.tokenizer
    invalid = torch.tensor(
        [
            tokenizer.encode(
                'root = Callout("column", ":title", ":body")',
                placeholders=[":title", ":body"],
            )
        ],
        dtype=torch.long,
    )
    valid = torch.tensor(
        [
            tokenizer.encode(
                'root = Callout("warning", ":title", ":body")',
                placeholders=[":title", ":body"],
            )
        ],
        dtype=torch.long,
    )
    compact_valid = torch.tensor(
        [
            tokenizer.encode(
                'root = Stack([TextContent(":title")], "column")',
                placeholders=[":title"],
            )
        ],
        dtype=torch.long,
    )

    finalized_invalid = model._finalize_schema_enum_choices(
        invalid, [[":title", ":body"]]
    )
    finalized_valid = model._finalize_schema_enum_choices(valid, [[":title", ":body"]])
    finalized_compact_valid = model._finalize_schema_enum_choices(
        compact_valid, [[":title"]]
    )

    assert (
        model._decode_openui(finalized_invalid[0], placeholders=[":title", ":body"])
        == 'root = Callout("info", ":title", ":body")'
    )
    assert torch.equal(finalized_valid, valid)
    assert torch.equal(finalized_compact_valid, compact_valid)


def test_choice_tokenizer_rejects_open_vocabulary_enum_literal() -> None:
    from slm_training.dsl.language_contract import OutputContractError

    model = _model(output_tokenizer="choice")
    tokenizer = model.tokenizer

    with pytest.raises(OutputContractError, match="forbids free-form strings"):
        tokenizer.encode(
            'root = Slider(":notify", "tet", 1, 1)', placeholders=[":notify"]
        )


def test_schema_opaque_bias_penalizes_slots_only_for_optional_empty_schema() -> None:
    from slm_training.dsl.production_codec import CLOSE, OPEN_PREFIX
    from slm_training.models.choice_tokenizer import ChoiceDecodeState

    model = _model(output_tokenizer="choice", schema_opaque_decode_weight=4.0)
    tokenizer = model.tokenizer
    state = ChoiceDecodeState(tokenizer, slot_count=2)
    assert state.advance_id(tokenizer.token_to_id[f"{OPEN_PREFIX}Button"])
    slot_id = tokenizer.sym_id(1)
    close_id = tokenizer.token_to_id[CLOSE]
    scores = torch.zeros(2)

    assert model._schema_opaque_bias(state, (slot_id, close_id), scores) is None
    assert state.advance_id(tokenizer.sym_id(0))
    bias = model._schema_opaque_bias(state, (slot_id, close_id), scores)

    assert bias is not None
    assert bias.tolist() == [-4.0, 0.0]


def test_choice_tokenizer_has_no_dynamic_string_tokens() -> None:
    tokenizer = _model(output_tokenizer="choice").tokenizer

    assert "LIT_STR" not in tokenizer.token_to_id
    assert '#""' not in tokenizer.token_to_id


def test_schema_enum_close_bias_rewards_only_optional_enum_close() -> None:
    from slm_training.dsl.production_codec import CLOSE, LIT_PREFIX, OPEN_PREFIX
    from slm_training.models.choice_tokenizer import ChoiceDecodeState

    model = _model(
        output_tokenizer="choice",
        schema_enum_close_decode_weight=4.0,
    )
    tokenizer = model.tokenizer
    state = ChoiceDecodeState(tokenizer, slot_count=2)
    assert state.advance_id(tokenizer.token_to_id[f"{OPEN_PREFIX}Button"])
    slot_id = tokenizer.sym_id(1)
    close_id = tokenizer.token_to_id[CLOSE]
    scores = torch.zeros(2)

    assert model._schema_enum_close_bias(state, (slot_id, close_id), scores) is None
    assert state.advance_id(tokenizer.sym_id(0))
    assert state.advance_id(tokenizer.token_to_id[f"{LIT_PREFIX}null"])
    bias = model._schema_enum_close_bias(state, (slot_id, close_id), scores)

    assert bias is not None
    assert bias.tolist() == [0.0, 4.0]


def test_schema_opaque_close_bias_rewards_only_optional_empty_schema_close() -> None:
    from slm_training.dsl.production_codec import CLOSE, OPEN_PREFIX
    from slm_training.models.choice_tokenizer import ChoiceDecodeState

    model = _model(
        output_tokenizer="choice",
        schema_opaque_close_decode_weight=4.0,
    )
    tokenizer = model.tokenizer
    state = ChoiceDecodeState(tokenizer, slot_count=2)
    assert state.advance_id(tokenizer.token_to_id[f"{OPEN_PREFIX}Button"])
    slot_id = tokenizer.sym_id(1)
    close_id = tokenizer.token_to_id[CLOSE]
    scores = torch.zeros(2)

    assert model._schema_opaque_close_bias(state, (slot_id, close_id), scores) is None
    assert state.advance_id(tokenizer.sym_id(0))
    bias = model._schema_opaque_close_bias(state, (slot_id, close_id), scores)

    assert bias is not None
    assert bias.tolist() == [0.0, 4.0]


def test_schema_role_slot_bias_prefers_active_content_property_owner() -> None:
    from slm_training.dsl.production_codec import OPEN_PREFIX
    from slm_training.models.choice_tokenizer import ChoiceDecodeState

    model = _model(output_tokenizer="choice", schema_role_slot_decode_weight=4.0)
    tokenizer = model.tokenizer
    state = ChoiceDecodeState(tokenizer, slot_count=3)
    assert state.advance_id(tokenizer.token_to_id[f"{OPEN_PREFIX}Input"])
    assert state.advance_id(tokenizer.sym_id(0))
    name_id = tokenizer.sym_id(0)
    email_id = tokenizer.sym_id(1)
    create_id = tokenizer.sym_id(2)
    bias = model._schema_role_slot_bias(
        state,
        (name_id, email_id, create_id),
        torch.zeros(3),
        [":auth.name", ":auth.email", ":auth.create"],
        {
            ":auth.name": ("Input",),
            ":auth.email": ("Input",),
            ":auth.create": ("Button",),
        },
    )

    assert bias is not None
    assert bias.tolist() == [4.0, 4.0, 0.0]


def test_schema_role_slot_bias_margin_floors_only_bound_unused_role() -> None:
    from slm_training.dsl.production_codec import OPEN_PREFIX
    from slm_training.models.choice_tokenizer import ChoiceDecodeState

    model = _model(output_tokenizer="choice", schema_role_slot_decode_weight=8.0)
    tokenizer = model.tokenizer
    state = ChoiceDecodeState(tokenizer, slot_count=2)
    assert state.advance_id(tokenizer.token_to_id[f"{OPEN_PREFIX}Button"])
    alt_id = tokenizer.sym_id(0)
    cta_id = tokenizer.sym_id(1)
    args = (
        state,
        (alt_id, cta_id),
        torch.tensor([20.0, 1.0]),
        [":gallery.alt", ":gallery.cta"],
        {
            ":gallery.alt": ("ImageGallery",),
            ":gallery.cta": ("Button",),
        },
    )

    bias = model._schema_role_slot_bias(
        *args,
        prefix=[],
        role_bindings={"Button": (":gallery.cta",)},
    )

    assert bias is not None
    assert bias.tolist() == [0.0, 27.0]
    assert (
        model._schema_role_slot_bias(
            *args,
            prefix=[cta_id],
            role_bindings={"Button": (":gallery.cta",)},
        )
        is None
    )


def test_schema_role_slot_bias_matches_bound_positional_property() -> None:
    from slm_training.data.quality import semantic_role_candidates
    from slm_training.dsl.production_codec import OPEN_PREFIX
    from slm_training.models.choice_tokenizer import ChoiceDecodeState

    model = _model(output_tokenizer="choice", schema_role_slot_decode_weight=8.0)
    tokenizer = model.tokenizer
    state = ChoiceDecodeState(tokenizer, slot_count=2)
    assert state.advance_id(tokenizer.token_to_id[f"{OPEN_PREFIX}CardHeader"])
    slots = [":hero.title", ":hero.subtitle"]
    candidates = semantic_role_candidates(slots, ["CardHeader"])
    title_id, subtitle_id = (tokenizer.sym_id(index) for index in range(2))

    title_bias = model._schema_role_slot_bias(
        state,
        (title_id, subtitle_id),
        torch.zeros(2),
        slots,
        candidates,
        prefix=[],
        role_bindings={"CardHeader": tuple(slots)},
    )
    assert title_bias is not None
    assert title_bias.tolist() == [8.0, 0.0]

    assert state.advance_id(title_id)
    subtitle_bias = model._schema_role_slot_bias(
        state,
        (title_id, subtitle_id),
        torch.zeros(2),
        slots,
        candidates,
        prefix=[title_id],
        role_bindings={"CardHeader": tuple(slots)},
    )
    assert subtitle_bias is not None
    assert subtitle_bias.tolist() == [0.0, 8.0]


def test_schema_role_slot_bias_prefers_active_typed_object_property() -> None:
    from slm_training.data.quality import semantic_role_candidates
    from slm_training.dsl.production_codec import NAME_PREFIX, OPEN_PREFIX
    from slm_training.models.choice_tokenizer import ChoiceDecodeState

    model = _model(output_tokenizer="choice", schema_role_slot_decode_weight=4.0)
    tokenizer = model.tokenizer
    state = ChoiceDecodeState(tokenizer, slot_count=3)
    for token in (
        f"{OPEN_PREFIX}ImageGallery",
        "[",
        "{",
        f"{NAME_PREFIX}src",
    ):
        assert state.advance_id(tokenizer.token_to_id[token])
    slots = [":gallery.img", ":gallery.alt", ":gallery.caption"]
    candidates = semantic_role_candidates(slots, ["ImageGallery"])
    slot_ids = tuple(tokenizer.sym_id(index) for index in range(3))

    bias = model._schema_role_slot_bias(
        state,
        slot_ids,
        torch.zeros(3),
        slots,
        candidates,
        semantic_role_properties={
            ":gallery.img": ("img", "src"),
            ":gallery.alt": ("alt",),
            ":gallery.caption": ("caption",),
        },
    )

    assert candidates == {
        ":gallery.alt": ("ImageGallery",),
        ":gallery.caption": ("ImageGallery",),
        ":gallery.img": ("ImageGallery",),
    }
    assert bias is not None
    assert bias.tolist() == [4.0, 0.0, 0.0]


def test_semantic_role_candidates_map_visible_content_aliases_to_schema() -> None:
    from slm_training.data.quality import semantic_role_candidates

    candidates = semantic_role_candidates(
        [":modal.title", ":modal.body", ":modal.confirm"],
        ["Modal", "TextContent", "Button"],
    )

    assert candidates == {
        ":modal.body": ("TextContent",),
        ":modal.confirm": ("Button",),
        ":modal.title": ("Modal", "TextContent"),
    }


def test_semantic_role_candidates_map_refresh_action_to_button_label() -> None:
    from slm_training.data.quality import semantic_role_candidates

    candidates = semantic_role_candidates(
        [":dashboard.refresh"],
        ["Button", "TextContent"],
    )

    assert candidates == {":dashboard.refresh": ("Button",)}


def test_semantic_role_candidates_map_display_heading_and_kicker_to_text() -> None:
    from slm_training.data.quality import semantic_role_candidates

    candidates = semantic_role_candidates(
        [":hero.kicker", ":callout.heading"],
        ["Callout", "CardHeader", "TextContent"],
    )

    assert candidates == {
        ":callout.heading": ("TextContent",),
        ":hero.kicker": ("TextContent",),
    }


def test_semantic_role_candidates_map_numbered_tabs_to_trigger() -> None:
    from slm_training.data.quality import semantic_role_candidates

    candidates = semantic_role_candidates(
        [":tabs.overview", ":tabs.tab1", ":tabs.tab2"],
        ["TabItem", "TextContent"],
    )

    assert candidates == {
        ":tabs.overview": ("TextContent",),
        ":tabs.tab1": ("TabItem",),
        ":tabs.tab2": ("TabItem",),
    }


def test_joint_role_candidates_require_distinct_schema_properties() -> None:
    slots = [":gallery.hint.title", ":gallery.hint.body"]
    candidates = TwoTowerModel._semantic_role_joint_candidates(
        slots, ["Callout", "TextCallout", "TextContent"]
    )

    assert candidates[tuple(sorted(slots))] == ("Callout", "TextCallout")


def test_joint_role_candidates_partition_larger_namespace_by_specificity() -> None:
    slots = [
        ":hero.body",
        ":hero.kicker",
        ":hero.subtitle",
        ":hero.title",
    ]
    candidates = TwoTowerModel._semantic_role_joint_candidates(
        slots, ["Callout", "CardHeader", "TextContent"]
    )

    assert candidates == {
        (":hero.subtitle", ":hero.title"): ("CardHeader",),
    }


def test_role_obligations_use_one_joint_schema_carrier() -> None:
    counts, bindings = TwoTowerModel._semantic_plan_role_obligations(
        Counter({"ImageGallery": 1}),
        {
            ":gallery.hint.title": ("Callout", "TextContent"),
            ":gallery.hint.body": ("Callout", "TextContent"),
        },
    )

    assert counts == Counter({"ImageGallery": 1, "Callout": 1})
    assert bindings["Callout"] == (
        ":gallery.hint.body",
        ":gallery.hint.title",
    )


def test_role_obligations_partition_hero_roles_into_schema_carriers() -> None:
    counts, bindings = TwoTowerModel._semantic_plan_role_obligations(
        Counter({"Card": 1, "Stack": 1}),
        {
            ":hero.body": ("Callout", "TextContent"),
            ":hero.kicker": ("TextContent",),
            ":hero.subtitle": ("CardHeader",),
            ":hero.title": ("Callout", "CardHeader", "TextContent"),
        },
    )

    assert counts == Counter({"TextContent": 2, "Card": 1, "Stack": 1, "CardHeader": 1})
    assert bindings == {
        "CardHeader": (":hero.subtitle", ":hero.title"),
        "TextContent": (":hero.body", ":hero.kicker"),
    }


def test_role_obligations_plan_unique_non_house_style_carriers() -> None:
    counts, bindings = TwoTowerModel._semantic_plan_role_obligations(
        Counter({"Tabs": 1}),
        {
            ":tabs.tab1": ("TabItem",),
            ":tabs.tab2": ("TabItem",),
        },
    )

    assert counts == Counter({"TabItem": 2, "Tabs": 1})
    assert bindings == {
        "TabItem": (":tabs.tab1", ":tabs.tab2"),
    }


def test_role_obligations_disambiguate_visible_role_with_public_enum() -> None:
    counts, bindings = TwoTowerModel._semantic_plan_role_obligations(
        Counter({"Button": 1, "Stack": 1}),
        {
            ":form.email": ("DatePicker", "Input", "Select", "TextArea"),
            ":form.submit": ("Button",),
        },
    )

    assert counts == Counter({"Button": 1, "Stack": 1, "Input": 1})
    assert bindings == {
        "Button": (":form.submit",),
        "Input": (":form.email",),
    }


def test_role_obligations_replace_exhausted_joint_carrier() -> None:
    counts, bindings = TwoTowerModel._semantic_plan_role_obligations(
        Counter({"Button": 1, "Stack": 1}),
        {
            ":held.form.title": (
                "Callout",
                "CardHeader",
                "Label",
                "Modal",
                "StepsItem",
                "Tag",
                "TextCallout",
                "TextContent",
            ),
            ":held.form.email": (
                "CheckBoxGroup",
                "DatePicker",
                "Input",
                "RadioGroup",
                "Select",
                "Slider",
                "TextArea",
            ),
            ":held.form.hint.title": (
                "Callout",
                "CardHeader",
                "Label",
                "Modal",
                "StepsItem",
                "Tag",
                "TextCallout",
                "TextContent",
            ),
            ":held.form.hint.body": (
                "Callout",
                "CheckBoxItem",
                "Label",
                "RadioItem",
                "SwitchItem",
                "Tag",
                "TextCallout",
                "TextContent",
            ),
            ":held.form.submit": (
                "Button",
                "CheckBoxItem",
                "Col",
                "FormControl",
                "RadioItem",
                "SelectItem",
                "Slider",
                "SwitchItem",
            ),
        },
        {
            ":held.form.title": ("Callout", "TextContent"),
            ":held.form.email": (),
            ":held.form.hint.title": ("Callout", "TextContent"),
            ":held.form.hint.body": ("Callout", "TextContent"),
            ":held.form.submit": ("Button",),
        },
    )

    assert counts == Counter(
        {"Button": 1, "Callout": 1, "Input": 1, "Stack": 1, "TextContent": 1}
    )
    assert bindings == {
        "Button": (":held.form.submit",),
        "Callout": (":held.form.hint.body", ":held.form.hint.title"),
        "Input": (":held.form.email",),
        "TextContent": (":held.form.title",),
    }


def test_role_obligations_abstain_when_capacity_has_no_property_match() -> None:
    assert isinstance(
        TwoTowerModel.__dict__["_semantic_plan_role_obligations"], staticmethod
    )
    counts, bindings = TwoTowerModel._semantic_plan_role_obligations(
        Counter({"Button": 1, "Input": 2}),
        {
            ":auth.name": ("CheckBoxItem", "Input"),
            ":auth.email": ("Input",),
            ":auth.create": ("Button", "CheckBoxItem"),
        },
    )

    assert counts == Counter({"Button": 1, "Input": 2})
    assert bindings == {
        "Button": (":auth.create",),
        "Input": (":auth.name", ":auth.email"),
    }


def test_role_obligations_disambiguate_children_from_planned_parent_schema() -> None:
    from slm_training.data.quality import semantic_role_candidates
    from slm_training.dsl.lang_core import library_schema

    slots = [":tabs.tab1", ":tabs.tab2"]
    candidates = semantic_role_candidates(
        slots,
        sorted(library_schema()["$defs"]),
    )

    counts, bindings = TwoTowerModel._semantic_plan_role_obligations(
        Counter({"Tabs": 1}),
        candidates,
    )

    assert candidates == {
        ":tabs.tab1": ("AccordionItem", "TabItem"),
        ":tabs.tab2": ("AccordionItem", "TabItem"),
    }
    assert counts == Counter({"TabItem": 2, "Tabs": 1})
    assert bindings == {"TabItem": (":tabs.tab1", ":tabs.tab2")}


def test_prompt_semantic_plan_recognizes_one_plural_family_container() -> None:
    from slm_training.models.template_fill import prompt_semantic_plan

    plan = prompt_semantic_plan(
        "Two-tab panel with an introductory heading and details content."
    )

    assert plan is not None
    assert [slot.component_family for slot in plan.role_slots] == ["Tabs"]


def test_prompt_semantic_plan_recognizes_public_group_from_unique_base() -> None:
    from slm_training.models.template_fill import prompt_semantic_plan

    plan = prompt_semantic_plan("Settings list with a switch and a slider.")

    assert plan is not None
    assert [slot.component_family for slot in plan.role_slots] == [
        "Slider",
        "SwitchGroup",
    ]


def test_prompt_semantic_plan_does_not_require_likeness_modifier() -> None:
    from slm_training.models.template_fill import prompt_semantic_plan

    plan = prompt_semantic_plan("Form-like stack with title text and submit button.")

    assert plan is not None
    assert [slot.component_family for slot in plan.role_slots] == ["Button", "Stack"]


def test_prompt_semantic_plan_bias_reaches_root_and_bound_components() -> None:
    from slm_training.data.semantic_plan import OpenUISemanticPlanCompiler
    from slm_training.models.template_fill import prompt_semantic_plan

    model = _model(
        output_tokenizer="choice",
        semantic_plan_decode_weight=3.0,
    )
    tokenizer = model.tokenizer
    component_ids = model._component_inventory_token_ids()
    actions = [
        str(tokenizer.id_to_token[token_id]).removeprefix("+")
        for token_id in component_ids
    ]
    plan = prompt_semantic_plan("Image gallery block with caption text underneath.")
    features = OpenUISemanticPlanCompiler().annotate_actions(None, actions, plan)
    model._semantic_plan_action_scores = [
        {
            token_id: feature.plan_confidence
            for token_id, feature in zip(component_ids, features, strict=True)
            if feature.component_family_compatible
        }
    ]
    gallery_id = tokenizer.token_to_id["+ImageGallery"]
    slider_id = tokenizer.token_to_id["+Slider"]
    candidates = (gallery_id, slider_id)

    root_bias = model._semantic_plan_bias(
        0, candidates, ("component_root", "component_root")
    )
    bound_bias = model._semantic_plan_bias(
        0, candidates, ("component_bound", "component_bound")
    )
    lexer_bias = model._semantic_plan_bias(0, candidates, ("component", "component"))

    assert root_bias is not None and root_bias.tolist() == [3.0, 0.0]
    assert bound_bias is not None and bound_bias.tolist() == [3.0, 0.0]
    assert lexer_bias is not None and lexer_bias.tolist() == [3.0, 0.0]


def test_prompt_semantic_plan_reaches_lexer_compiler(monkeypatch) -> None:
    model = _model(
        semantic_plan_decode_weight=4.0,
        semantic_plan_margin_decode_weight=2.0,
    )
    card_id = model.tokenizer.token_to_id["Card"]

    def decode(_ctx, _ctx_pad, length: int) -> torch.Tensor:
        assert model._semantic_plan_action_scores
        assert card_id in model._semantic_plan_action_scores[0]
        return torch.full((1, length), model.tokenizer.eos_id, dtype=torch.long)

    monkeypatch.setattr(model, "_greedy_ltr_decode_batch", decode)
    monkeypatch.setattr(model, "_decode_ids", lambda _ids: "root = Stack([])")
    monkeypatch.setattr(model, "_ensure_valid_openui", lambda text, *_a, **_k: text)

    assert model.generate_batch_requests(
        [GenerationRequest(prompt="Card layout.")]
    ) == ["root = Stack([])"]


@pytest.mark.parametrize("path_kind", ["component", "component_bound"])
def test_prompt_semantic_plan_ranks_lexer_tree_paths(path_kind: str) -> None:
    from types import SimpleNamespace

    model = _model(
        semantic_plan_decode_weight=4.0,
        semantic_plan_margin_decode_weight=2.0,
    )
    tokenizer = model.tokenizer
    card = tokenizer.token_to_id["Card"]
    text = tokenizer.token_to_id["TextContent"]
    prefix = [tokenizer.bos_id, *tokenizer.encode("root=Stack([", add_special=False)]
    paths = (
        CompletionPath((card, tokenizer.token_to_id["("]), path_kind),
        CompletionPath((text, tokenizer.token_to_id["("]), path_kind),
    )
    model._semantic_plan_action_scores = [{card: 1.0}]
    model._semantic_plan_action_counts = [{card: 1}]
    ctx, ctx_pad = model._encode_context(["Card layout."])

    with collect_decode_stats() as stats:
        selected = model._select_compiler_path(
            prefix,
            paths,
            ctx,
            ctx_pad,
            24,
            tree=True,
            state=SimpleNamespace(section_types=(), frames=()),
        )

    assert selected == paths[0].token_ids
    assert stats.semantic_plan_applications == 1

    root_paths = tuple(
        CompletionPath(path.token_ids, "component_root") for path in paths
    )
    with collect_decode_stats() as root_stats:
        model._select_compiler_path(
            prefix,
            root_paths,
            ctx,
            ctx_pad,
            24,
            tree=True,
            state=SimpleNamespace(section_types=(), frames=()),
        )
    assert root_stats.semantic_plan_applications == 0

    covered = model._semantic_plan_covered_counts(
        SimpleNamespace(),
        tokenizer.encode("root = Stack([b1])\nb1 = Card([])", add_special=True),
        {"Card": card, "TextContent": text},
    )
    assert covered == Counter({card: 1})


def test_prompt_semantic_plan_keeps_repeated_lexer_families_as_siblings() -> None:
    model = _model(
        semantic_plan_decode_weight=6.0,
        semantic_plan_margin_decode_weight=2.0,
    )
    tokenizer = model.tokenizer
    card = tokenizer.token_to_id["Card"]
    text = tokenizer.token_to_id["TextContent"]
    state = make_grammar_state()
    prefix = [
        tokenizer.bos_id,
        *tokenizer.encode("root = Stack([Card([", add_special=False),
    ]
    for token_id in prefix[1:]:
        state.advance_token(tokenizer, token_id)
    model._semantic_plan_action_scores = [{card: 1.0}]
    model._semantic_plan_action_counts = [{card: 5}]
    model._semantic_plan_outer_groups = [None]

    bias = model._semantic_plan_bias(
        0,
        (card, text),
        ("component", "component"),
        state,
        prefix,
        torch.tensor([8.0, 1.0]),
    )

    assert bias is not None
    assert (torch.tensor([8.0, 1.0]) + bias).argmax().item() == 1


def test_prompt_semantic_plan_keeps_parent_open_for_remaining_siblings() -> None:
    model = _model(
        semantic_plan_decode_weight=6.0,
        semantic_plan_margin_decode_weight=2.0,
    )
    tokenizer = model.tokenizer
    card = tokenizer.token_to_id["Card"]
    comma = tokenizer.token_to_id[","]
    close = tokenizer.token_to_id["]"]
    prefix = [
        tokenizer.bos_id,
        *tokenizer.encode(
            'root = Stack([Card([TextContent(":sym0")])',
            add_special=False,
        ),
    ]
    state = make_grammar_state()
    for token_id in prefix[1:]:
        state.advance_token(tokenizer, token_id)
    model._semantic_plan_action_scores = [{card: 1.0}]
    model._semantic_plan_action_counts = [{card: 5}]
    model._semantic_plan_outer_groups = [None]

    scores = torch.tensor([1.0, 4.0])
    bias = model._semantic_plan_bias(
        0,
        (comma, close),
        ("structural", "structural"),
        state,
        prefix,
        scores,
    )

    assert bias is not None
    assert (scores + bias).argmax().item() == 0


def test_prompt_semantic_plan_closes_array_after_opaque_coverage() -> None:
    model = _model(
        semantic_plan_decode_weight=4.0,
        semantic_plan_margin_decode_weight=2.0,
    )
    tokenizer = model.tokenizer
    card = tokenizer.token_to_id["Card"]
    comma = tokenizer.token_to_id[","]
    close = tokenizer.token_to_id["]"]
    prefix = [
        tokenizer.bos_id,
        *tokenizer.encode(
            'root = Stack([Card([TextContent(":slot_0")])',
            add_special=False,
        ),
    ]
    state = make_grammar_state()
    for token_id in prefix[1:]:
        state.advance_token(tokenizer, token_id)
    model._semantic_plan_action_scores = [{card: 1.0}]
    model._semantic_plan_action_counts = [{card: 1}]
    model._slot_contracts = [[":slot_0"]]

    scores = torch.tensor([8.0, 1.0])
    bias = model._semantic_plan_bias(
        0,
        (comma, close),
        ("structural", "structural"),
        state,
        prefix,
        scores,
    )

    assert bias is not None
    assert (scores + bias).argmax().item() == 1

    model._slot_contracts = [[":slot_0", ":slot_1"]]
    assert (
        model._semantic_plan_bias(
            0,
            (comma, close),
            ("structural", "structural"),
            state,
            prefix,
            scores,
        )
        is None
    )


def test_prompt_semantic_plan_does_not_continue_inside_repeated_family() -> None:
    model = _model(
        semantic_plan_decode_weight=6.0,
        semantic_plan_margin_decode_weight=2.0,
    )
    tokenizer = model.tokenizer
    card = tokenizer.token_to_id["Card"]
    comma = tokenizer.token_to_id[","]
    close = tokenizer.token_to_id["]"]
    prefix = [
        tokenizer.bos_id,
        *tokenizer.encode(
            'root = Stack([Card([TextContent(":sym0")',
            add_special=False,
        ),
    ]
    state = make_grammar_state()
    for token_id in prefix[1:]:
        state.advance_token(tokenizer, token_id)
    model._semantic_plan_action_scores = [{card: 1.0}]
    model._semantic_plan_action_counts = [{card: 5}]
    model._semantic_plan_outer_groups = [None]

    assert (
        model._semantic_plan_bias(
            0,
            (comma, close),
            ("structural", "structural"),
            state,
            prefix,
            torch.tensor([1.0, 4.0]),
        )
        is None
    )


def test_lexer_semantic_plan_margin_keeps_planned_typed_array_nonempty(
    monkeypatch,
) -> None:
    model = _model(
        semantic_plan_margin_decode_weight=2.0,
        semantic_plan_typed_array_item_margin_decode_weight=2.0,
    )
    tokenizer = model.tokenizer
    prefix = [
        tokenizer.bos_id,
        *tokenizer.encode('root = SwitchGroup(":group", [', add_special=False),
    ]
    state = make_grammar_state()
    for token_id in prefix[1:]:
        state.advance_token(tokenizer, token_id)
    binder_id = tokenizer.token_to_id["<BIND_1>"]
    group_id = tokenizer.token_to_id["SwitchGroup"]
    item_id = tokenizer.token_to_id["SwitchItem"]
    model._semantic_plan_action_counts = [{group_id: 1}]
    model._slot_contracts = [[":hero.title"]]

    bias = model._semantic_plan_typed_array_nonempty_bias(
        0,
        state,
        prefix,
        (binder_id, item_id),
        torch.tensor([9.0, 1.0]),
    )

    assert bias is not None
    assert bias.tolist() == [0.0, 10.0]

    monkeypatch.setattr(
        model,
        "_project_candidates",
        lambda _hidden, candidate_ids: torch.tensor(
            [9.0 if token_id == binder_id else 1.0 for token_id in candidate_ids]
        ),
    )
    ctx, ctx_pad = model._encode_context(["Card layout."])
    paths = (
        CompletionPath((binder_id,), "binder"),
        CompletionPath((item_id, tokenizer.token_to_id["("]), "component"),
    )
    selected = model._select_compiler_path(
        prefix,
        paths,
        ctx,
        ctx_pad,
        32,
        tree=True,
        state=state,
    )

    assert selected == paths[1].token_ids


def test_lexer_typed_item_margin_abstains_for_recursive_unique_item() -> None:
    model = _model(
        semantic_plan_margin_decode_weight=2.0,
        semantic_plan_typed_array_item_margin_decode_weight=2.0,
    )
    tokenizer = model.tokenizer
    prefix = [
        tokenizer.bos_id,
        *tokenizer.encode("root = Tabs([", add_special=False),
    ]
    state = make_grammar_state()
    for token_id in prefix[1:]:
        state.advance_token(tokenizer, token_id)
    binder_id = tokenizer.token_to_id["<BIND_1>"]
    tabs_id = tokenizer.token_to_id["Tabs"]
    item_id = tokenizer.token_to_id["TabItem"]
    model._semantic_plan_action_counts = [{tabs_id: 1}]
    model._slot_contracts = [[":tab", ":content"]]

    assert (
        model._semantic_plan_typed_array_nonempty_bias(
            0,
            state,
            prefix,
            (binder_id, item_id),
            torch.tensor([9.0, 1.0]),
        )
        is None
    )


def test_lexer_schema_types_prefer_recursive_unique_item() -> None:
    model = _model(
        compiler_schema_component_types=True,
        semantic_plan_margin_decode_weight=2.0,
    )
    tokenizer = model.tokenizer
    prefix = [
        tokenizer.bos_id,
        *tokenizer.encode("root = Tabs([", add_special=False),
    ]
    state = make_grammar_state()
    for token_id in prefix[1:]:
        state.advance_token(tokenizer, token_id)
    binder_id = tokenizer.token_to_id["<BIND_1>"]
    tabs_id = tokenizer.token_to_id["Tabs"]
    item_id = tokenizer.token_to_id["TabItem"]
    model._semantic_plan_action_counts = [{tabs_id: 1}]
    model._slot_contracts = [[":tab", ":content"]]

    bias = model._semantic_plan_typed_array_nonempty_bias(
        0,
        state,
        prefix,
        (binder_id, item_id),
        torch.tensor([9.0, 1.0]),
    )

    assert bias is not None
    assert bias.tolist() == [0.0, 10.0]


def test_lexer_schema_role_slot_bias_ranks_nested_compiler_edge(monkeypatch) -> None:
    model = _model(schema_role_slot_decode_weight=2.0)
    tokenizer = model.tokenizer
    title_id = tokenizer.sym_id(0)
    body_id = tokenizer.sym_id(1)
    prefix = [
        tokenizer.bos_id,
        *tokenizer.encode("root = ", add_special=False),
    ]
    state = make_grammar_state()
    for token_id in prefix[1:]:
        state.advance_token(tokenizer, token_id)
    model._slot_contracts = [[":hero.title", ":hero.body"]]
    model._semantic_role_candidates = [{}]
    model._semantic_role_properties = [
        {":hero.title": ("title",), ":hero.body": ("text",)}
    ]
    monkeypatch.setattr(
        model,
        "_project_candidates",
        lambda _hidden, candidate_ids: torch.tensor(
            [4.0 if token_id == body_id else 1.0 for token_id in candidate_ids]
        ),
    )
    card_header = tokenizer.token_to_id["CardHeader"]
    lpar = tokenizer.token_to_id["("]
    comma = tokenizer.token_to_id[","]
    paths = (
        CompletionPath((card_header, lpar, title_id, comma), "component_root"),
        CompletionPath((card_header, lpar, body_id, comma), "component_root"),
    )
    ctx, ctx_pad = model._encode_context(["Hero card."])

    selected = model._select_compiler_path(
        prefix,
        paths,
        ctx,
        ctx_pad,
        32,
        tree=True,
        slot_contract=model._slot_contracts[0],
        state=state,
    )

    assert selected == paths[0].token_ids


def test_lexer_semantic_role_ranks_compiler_component_family(monkeypatch) -> None:
    model = _model(semantic_role_decode_weight=2.0)
    tokenizer = model.tokenizer
    button = tokenizer.token_to_id["Button"]
    text = tokenizer.token_to_id["TextContent"]
    prefix = [tokenizer.bos_id, *tokenizer.encode("root = Stack([", add_special=False)]
    state = make_grammar_state()
    for token_id in prefix[1:]:
        state.advance_token(tokenizer, token_id)
    model._slot_contracts = [[":cta.label"]]
    model._semantic_role_candidates = [{":cta.label": ("Button",)}]
    monkeypatch.setattr(
        model,
        "_project_candidates",
        lambda _hidden, candidate_ids: torch.tensor(
            [2.0 if token_id == text else 1.0 for token_id in candidate_ids]
        ),
    )
    paths = (
        CompletionPath((button, tokenizer.token_to_id["("]), "component"),
        CompletionPath((text, tokenizer.token_to_id["("]), "component"),
    )
    ctx, ctx_pad = model._encode_context(["Single button."])

    selected = model._select_compiler_path(
        prefix,
        paths,
        ctx,
        ctx_pad,
        32,
        tree=True,
        slot_contract=model._slot_contracts[0],
        state=state,
    )

    assert selected == paths[0].token_ids


@pytest.mark.parametrize("tree", [False, True])
def test_lexer_slot_coverage_close_stops_after_declared_symbols(
    monkeypatch, tree: bool
) -> None:
    model = _model(slot_coverage_close_decode_weight=2.0)
    tokenizer = model.tokenizer
    close_id = tokenizer.token_to_id["]"]
    text = tokenizer.token_to_id["TextContent"]
    prefix = [tokenizer.bos_id, tokenizer.sym_id(0)]
    paths = (
        CompletionPath((close_id,), "grammar_rsqb_populated"),
        CompletionPath((text, tokenizer.token_to_id["("]), "component"),
    )
    monkeypatch.setattr(
        model,
        "_project_candidates",
        lambda _hidden, candidate_ids: torch.tensor(
            [5.0 if token_id == text else 1.0 for token_id in candidate_ids]
        ),
    )
    ctx, ctx_pad = model._encode_context(["Single label."])

    selected = model._select_compiler_path(
        prefix,
        paths,
        ctx,
        ctx_pad,
        32,
        tree=tree,
        slot_contract=[":cta.label"],
        state=make_grammar_state(),
    )

    assert selected == paths[0].token_ids

    incomplete = model._select_compiler_path(
        [tokenizer.bos_id],
        paths,
        ctx,
        ctx_pad,
        32,
        tree=tree,
        slot_contract=[":cta.label"],
        state=make_grammar_state(),
    )

    assert incomplete == paths[1].token_ids


@pytest.mark.parametrize("tree", [False, True])
def test_lexer_required_slot_margin_uses_missing_visible_symbol(
    monkeypatch, tree: bool
) -> None:
    model = _model(required_slot_margin_decode_weight=2.0)
    tokenizer = model.tokenizer
    title_id = tokenizer.sym_id(0)
    body_id = tokenizer.sym_id(1)
    prefix = [
        tokenizer.bos_id,
        *tokenizer.encode("root = Stack([TextContent(", add_special=False),
        body_id,
        *tokenizer.encode("), TextContent(", add_special=False),
    ]
    state = make_grammar_state()
    for token_id in prefix[1:]:
        state.advance_token(tokenizer, token_id)
    slot_contract = [":title", ":body"]
    paths = (
        CompletionPath((title_id,), "slot"),
        CompletionPath((body_id,), "slot"),
    )
    monkeypatch.setattr(
        model,
        "_project_candidates",
        lambda _hidden, candidate_ids: torch.tensor(
            [1.0 if token_id == title_id else 4.0 for token_id in candidate_ids]
        ),
    )
    ctx, ctx_pad = model._encode_context(["Two text fields."])

    with collect_decode_stats() as stats:
        selected = model._select_compiler_path(
            prefix,
            paths,
            ctx,
            ctx_pad,
            48,
            tree=tree,
            slot_contract=slot_contract,
            state=state,
        )

    assert selected == paths[0].token_ids
    assert stats.required_slot_margin_applications == 1
    assert stats.required_slot_margin_choice_changes == 1


@pytest.mark.parametrize("tree", [False, True])
def test_lexer_schema_role_slot_continues_to_missing_bound_property(
    monkeypatch, tree: bool
) -> None:
    model = _model(schema_role_slot_decode_weight=2.0)
    tokenizer = model.tokenizer
    title_id = tokenizer.sym_id(0)
    subtitle_id = tokenizer.sym_id(1)
    prefix = [
        tokenizer.bos_id,
        *tokenizer.encode("root = Card([CardHeader(", add_special=False),
        title_id,
    ]
    state = make_grammar_state()
    for token_id in prefix[1:]:
        state.advance_token(tokenizer, token_id)
    comma_id = tokenizer.token_to_id[","]
    close_id = tokenizer.token_to_id[")"]
    paths = (
        CompletionPath((comma_id,), "structural"),
        CompletionPath((close_id,), "structural"),
    )
    model._semantic_plan_role_bindings = [
        {"CardHeader": (":hero.title", ":hero.subtitle")}
    ]
    model._semantic_role_properties = [
        {
            ":hero.title": ("text", "title"),
            ":hero.subtitle": ("subtitle",),
        }
    ]
    monkeypatch.setattr(
        model,
        "_project_candidates",
        lambda _hidden, candidate_ids: torch.tensor(
            [1.0 if token_id == comma_id else 4.0 for token_id in candidate_ids]
        ),
    )
    ctx, ctx_pad = model._encode_context(["Hero card."])

    selected = model._select_compiler_path(
        prefix,
        paths,
        ctx,
        ctx_pad,
        48,
        tree=tree,
        slot_contract=[":hero.title", ":hero.subtitle"],
        state=state,
    )

    assert subtitle_id not in prefix
    assert selected == paths[0].token_ids


def test_prompt_semantic_plan_bias_is_neutral_without_prompt_mentions() -> None:
    from slm_training.data.semantic_plan import OpenUISemanticPlanCompiler
    from slm_training.models.template_fill import prompt_semantic_plan

    model = _model(
        output_tokenizer="choice",
        semantic_plan_decode_weight=3.0,
    )
    assert prompt_semantic_plan("Build a polished responsive experience.") is None
    features = OpenUISemanticPlanCompiler().annotate_actions(
        None, ["Card", "Button"], None
    )
    assert all(feature.plan_confidence == 0.0 for feature in features)
    model._semantic_plan_action_scores = [{}]
    assert (
        model._semantic_plan_bias(
            0,
            (model.tokenizer.token_to_id["+Card"],),
            ("component_root",),
        )
        is None
    )


def test_prompt_semantic_plan_preserves_repeated_authored_component_mentions() -> None:
    from slm_training.models.template_fill import prompt_semantic_plan

    plan = prompt_semantic_plan(
        "Sign-up column with name input, email input, and create button.\n"
        "Components: Button, Input\n"
        "Semantic roles: auth(name -> Input, email -> Input, create -> Button)"
    )

    assert plan is not None
    assert [slot.component_family for slot in plan.role_slots] == [
        "Button",
        "Input",
        "Input",
    ]


def test_prompt_semantic_plan_preserves_modified_component_count() -> None:
    from slm_training.models.template_fill import prompt_semantic_plan

    plan = prompt_semantic_plan(
        "Dashboard with a status callout, two metric cards, and a refresh action."
    )

    assert plan is not None
    assert [slot.component_family for slot in plan.role_slots] == [
        "Callout",
        "Card",
        "Card",
        "Button",
    ]


def test_prompt_semantic_plan_preserves_count_across_multiple_modifiers() -> None:
    from slm_training.models.template_fill import prompt_semantic_plan

    plan = prompt_semantic_plan("Row of three equally important action buttons.")

    assert plan is not None
    assert [slot.component_family for slot in plan.role_slots] == [
        "Button",
        "Button",
        "Button",
    ]


def test_prompt_semantic_plan_does_not_cross_component_conjunction() -> None:
    from slm_training.models.template_fill import prompt_semantic_plan

    plan = prompt_semantic_plan("Place three cards and a button in a row.")

    assert plan is not None
    assert [slot.component_family for slot in plan.role_slots] == [
        "Button",
        "Card",
        "Card",
        "Card",
    ]


def test_prompt_semantic_plan_excludes_inline_role_metadata_from_counts() -> None:
    from slm_training.models.template_fill import prompt_semantic_plan

    plan = prompt_semantic_plan(
        "Build a column mobile layout with 5 cards, 1 text block "
        "(roles: card, text) using placeholders only."
    )

    assert plan is not None
    assert [slot.component_family for slot in plan.role_slots] == [
        "Card",
        "Card",
        "Card",
        "Card",
        "Card",
    ]


def test_prompt_semantic_plan_extracts_explicit_outer_group_topology() -> None:
    from slm_training.models.template_fill import prompt_semantic_plan

    plan = prompt_semantic_plan(
        "Place two cards around a separator, then put the group inside an outer card."
    )

    assert plan is not None
    assert plan.topology.parent_relation_candidates == (
        {
            "relation": "outer_group",
            "parent_role_id": "prompt_component_2",
            "parent_family": "Card",
            "group_family": "Stack",
            "sibling_role_ids": [
                "prompt_component_0",
                "prompt_component_3",
                "prompt_component_1",
            ],
            "sibling_families": ["Card", "Separator", "Card"],
            "direction": "column",
            "evidence": "group inside an outer card",
        },
    )


def test_prompt_semantic_plan_infers_button_from_action_semantics() -> None:
    from slm_training.models.template_fill import prompt_semantic_plan

    plan = prompt_semantic_plan("Modal dialog confirming a destructive delete action.")

    assert plan is not None
    assert [slot.component_family for slot in plan.role_slots] == [
        "Modal",
        "Button",
    ]


def test_prompt_semantic_plan_bias_targets_only_missing_family_instances() -> None:
    from types import SimpleNamespace

    model = _model(
        output_tokenizer="choice",
        semantic_plan_decode_weight=3.0,
    )
    tokenizer = model.tokenizer
    input_id = tokenizer.token_to_id["+Input"]
    button_id = tokenizer.token_to_id["+Button"]
    candidates = (input_id, button_id)
    kinds = ("component_bound", "component_bound")
    model._semantic_plan_action_scores = [{input_id: 1.0, button_id: 1.0}]
    model._semantic_plan_action_counts = [{input_id: 2, button_id: 1}]
    one_each = SimpleNamespace(section_types=["element:Input", "element:Button"])
    all_required = SimpleNamespace(
        section_types=["element:Input", "element:Input", "element:Button"]
    )

    remaining_bias = model._semantic_plan_bias(0, candidates, kinds, one_each)
    complete_bias = model._semantic_plan_bias(0, candidates, kinds, all_required)

    assert remaining_bias is not None
    assert remaining_bias.tolist() == [3.0, 0.0]
    assert complete_bias is None


def test_prompt_semantic_plan_bias_counts_nested_family_instances() -> None:
    from types import SimpleNamespace

    model = _model(output_tokenizer="choice", semantic_plan_decode_weight=3.0)
    tokenizer = model.tokenizer
    modal_id = tokenizer.token_to_id["+Modal"]
    text_id = tokenizer.token_to_id["+TextContent"]
    buttons_id = tokenizer.token_to_id["+Buttons"]
    button_id = tokenizer.token_to_id["+Button"]
    model._semantic_plan_action_scores = [
        {
            modal_id: 1.0,
            text_id: 1.0,
            button_id: 1.0,
        }
    ]
    model._semantic_plan_action_counts = [
        {
            modal_id: 1,
            text_id: 1,
            button_id: 1,
        }
    ]
    state = SimpleNamespace(section_types=["element:Modal"], frames=[])
    prefix = [
        tokenizer.bos_id,
        modal_id,
        text_id,
        buttons_id,
        button_id,
    ]

    assert (
        model._semantic_plan_bias(
            0,
            (button_id,),
            ("component_bound",),
            state,
            prefix,
        )
        is None
    )


def test_prompt_semantic_plan_margin_floors_only_missing_families() -> None:
    from types import SimpleNamespace

    model = _model(
        output_tokenizer="choice",
        semantic_plan_margin_decode_weight=2.0,
    )
    tokenizer = model.tokenizer
    card_id = tokenizer.token_to_id["+Card"]
    text_id = tokenizer.token_to_id["+TextContent"]
    button_id = tokenizer.token_to_id["+Button"]
    candidates = (card_id, text_id, button_id)
    kinds = ("component_bound", "component_bound", "component_bound")
    model._semantic_plan_action_scores = [{card_id: 1.0, button_id: 1.0}]
    model._semantic_plan_action_counts = [{card_id: 1, button_id: 1}]
    state = SimpleNamespace(section_types=["element:Button"])

    bias = model._semantic_plan_bias(
        0,
        candidates,
        kinds,
        state,
        candidate_scores=torch.tensor([-36.0, 30.0, 12.0]),
    )

    assert bias is not None
    assert bias.tolist() == [68.0, 0.0, 0.0]


def test_prompt_semantic_plan_orders_required_parent_before_child() -> None:
    from types import SimpleNamespace

    model = _model(
        output_tokenizer="choice",
        semantic_plan_decode_weight=4.0,
        semantic_plan_margin_decode_weight=2.0,
    )
    tokenizer = model.tokenizer
    form_id = tokenizer.token_to_id["+Form"]
    button_id = tokenizer.token_to_id["+Button"]
    stack_id = tokenizer.token_to_id["+Stack"]
    callout_id = tokenizer.token_to_id["+Callout"]
    candidates = (button_id, form_id, stack_id, callout_id)
    model._semantic_plan_action_scores = [{token_id: 1.0 for token_id in candidates}]
    model._semantic_plan_action_counts = [{token_id: 1 for token_id in candidates}]

    bias = model._semantic_plan_bias(
        0,
        candidates,
        ("component_bound",) * 4,
        SimpleNamespace(section_types=[], frames=[]),
        prefix=[tokenizer.bos_id],
        candidate_scores=torch.zeros(4),
    )

    assert model._schema_required_descendant_families("Form") >= {
        "Buttons",
        "Button",
        "FormControl",
    }
    assert "Button" not in model._schema_required_descendant_families("Stack")
    assert bias is not None
    assert bias.tolist() == [4.0, 6.0, 4.0, 4.0]

    after_form = model._semantic_plan_bias(
        0,
        (stack_id, callout_id),
        ("component_bound", "component_bound"),
        SimpleNamespace(section_types=["element:Form", "element:Button"], frames=[]),
        prefix=[tokenizer.bos_id, form_id, button_id],
        candidate_scores=torch.zeros(2),
    )
    assert after_form is not None
    assert after_form.tolist() == [4.0, 6.0]


def test_prompt_semantic_plan_seed_bias_applies_only_before_first_component() -> None:
    from types import SimpleNamespace

    model = _model(
        output_tokenizer="choice",
        semantic_plan_seed_decode_weight=5.0,
    )
    tokenizer = model.tokenizer
    card_id = tokenizer.token_to_id["+Card"]
    text_id = tokenizer.token_to_id["+TextContent"]
    candidates = (card_id, text_id)
    kinds = ("component_root", "component_root")
    model._semantic_plan_action_scores = [{card_id: 1.0}]

    initial = SimpleNamespace(section_types=[], frames=[object()])
    after_first = SimpleNamespace(section_types=["element:TextContent"], frames=[])

    initial_bias = model._semantic_plan_bias(0, candidates, kinds, initial)
    nested_bias = model._semantic_plan_bias(
        0, candidates, ("component_bound", "component_bound"), initial
    )
    later_bias = model._semantic_plan_bias(0, candidates, kinds, after_first)

    assert initial_bias is not None
    assert initial_bias.tolist() == [5.0, 0.0]
    assert nested_bias is None
    assert later_bias is None


def test_prompt_semantic_plan_seed_trace_records_score_decomposition() -> None:
    from types import SimpleNamespace

    from slm_training.models.decode_stats import DecodeStats

    model = _model(
        output_tokenizer="choice",
        semantic_plan_decode_weight=4.0,
        semantic_plan_seed_decode_weight=8.0,
    )
    tokenizer = model.tokenizer
    card_id = tokenizer.token_to_id["+Card"]
    text_id = tokenizer.token_to_id["+TextContent"]
    stats = DecodeStats()

    trace = model._record_semantic_plan_seed_trace(
        stats,
        row=2,
        position=3,
        state=SimpleNamespace(section_types=[]),
        candidate_ids=(card_id, text_id),
        candidate_kinds=("component_root", "component_root"),
        scores_before=torch.tensor([1.0, 5.0]),
        plan_bias=torch.tensor([12.0, 0.0]),
        scores_after=torch.tensor([13.0, 5.0]),
    )
    model._finalize_semantic_plan_trace(
        trace,
        candidate_ids=(card_id, text_id),
        scores=torch.tensor([12.0, 15.0]),
    )

    assert stats.constrained_selection_traces == [
        {
            "phase": "semantic_plan_seed",
            "row": 2,
            "position": 3,
            "before_token": "+TextContent",
            "chosen_token": "+Card",
            "choice_changed": True,
            "final_token": "+TextContent",
            "changed_after_plan": True,
            "seed_weight": 8.0,
            "semantic_plan_decode_weight": 4.0,
            "top_candidates": [
                {
                    "token": "+Card",
                    "kind": "component_root",
                    "score_before": 1.0,
                    "plan_bias": 12.0,
                    "score_after": 13.0,
                    "post_plan_bias": -1.0,
                    "final_score": 12.0,
                },
                {
                    "token": "+TextContent",
                    "kind": "component_root",
                    "score_before": 5.0,
                    "plan_bias": 0.0,
                    "score_after": 5.0,
                    "post_plan_bias": 10.0,
                    "final_score": 15.0,
                },
            ],
        }
    ]


def test_prompt_semantic_plan_missing_family_trace_records_remaining_counts() -> None:
    from types import SimpleNamespace

    from slm_training.models.decode_stats import DecodeStats

    model = _model(
        output_tokenizer="choice",
        semantic_plan_decode_weight=32.0,
    )
    tokenizer = model.tokenizer
    card_id = tokenizer.token_to_id["+Card"]
    text_id = tokenizer.token_to_id["+TextContent"]
    model._semantic_plan_action_counts = [{card_id: 1, text_id: 1}]
    stats = DecodeStats()

    trace = model._record_semantic_plan_missing_family_trace(
        stats,
        row=0,
        position=12,
        state=SimpleNamespace(section_types=["element:TextContent"]),
        candidate_ids=(card_id, text_id),
        candidate_kinds=("component_bound", "component_bound"),
        scores_before=torch.tensor([1.0, 5.0]),
        plan_bias=torch.tensor([32.0, 0.0]),
        scores_after=torch.tensor([33.0, 5.0]),
    )
    model._finalize_semantic_plan_trace(
        trace,
        candidate_ids=(card_id, text_id),
        scores=torch.tensor([30.0, 35.0]),
    )

    assert stats.constrained_selection_traces == [
        {
            "phase": "semantic_plan_missing_family",
            "row": 0,
            "position": 12,
            "emitted_families": ["TextContent"],
            "remaining_planned_families": {"Card": 1},
            "before_token": "+TextContent",
            "chosen_token": "+Card",
            "choice_changed": True,
            "final_token": "+TextContent",
            "changed_after_plan": True,
            "semantic_plan_decode_weight": 32.0,
            "semantic_plan_margin_decode_weight": 0.0,
            "planned_candidates": [
                {
                    "token": "+Card",
                    "kind": "component_bound",
                    "score_before": 1.0,
                    "plan_bias": 32.0,
                    "score_after": 33.0,
                    "post_plan_bias": -3.0,
                    "final_score": 30.0,
                }
            ],
            "top_candidates": [
                {
                    "token": "+Card",
                    "kind": "component_bound",
                    "score_before": 1.0,
                    "plan_bias": 32.0,
                    "score_after": 33.0,
                    "post_plan_bias": -3.0,
                    "final_score": 30.0,
                },
                {
                    "token": "+TextContent",
                    "kind": "component_bound",
                    "score_before": 5.0,
                    "plan_bias": 0.0,
                    "score_after": 5.0,
                    "post_plan_bias": 30.0,
                    "final_score": 35.0,
                },
            ],
        }
    ]


def test_prompt_semantic_plan_root_trace_records_verified_target_scores() -> None:
    from types import SimpleNamespace

    from slm_training.models.decode_stats import DecodeStats

    model = _model(
        output_tokenizer="choice",
        semantic_plan_root_decode_weight=8.0,
    )
    tokenizer = model.tokenizer
    stack_id = tokenizer.token_to_id["+Stack"]
    text_id = tokenizer.token_to_id["+TextContent"]
    stats = DecodeStats()

    trace = model._record_semantic_plan_root_trace(
        stats,
        row=0,
        position=20,
        state=SimpleNamespace(
            section_types=["element:Card", "element:Callout"],
            frames=[],
            mode="structural",
        ),
        candidate_ids=(stack_id, text_id),
        scores_before=torch.tensor([1.0, 14.0]),
        root_bias=torch.tensor([8.0, 0.0]),
        scores_after=torch.tensor([9.0, 14.0]),
    )
    model._finalize_semantic_plan_trace(
        trace,
        candidate_ids=(stack_id, text_id),
        scores=torch.tensor([9.0, 15.0]),
    )

    assert trace is not None
    assert trace["phase"] == "semantic_plan_root"
    assert trace["emitted_families"] == ["Card", "Callout"]
    assert trace["before_token"] == "+TextContent"
    assert trace["chosen_token"] == "+TextContent"
    assert trace["final_token"] == "+TextContent"
    assert trace["planned_candidates"] == [
        {
            "token": "+Stack",
            "score_before": 1.0,
            "plan_bias": 8.0,
            "score_after": 9.0,
            "post_plan_bias": 0.0,
            "final_score": 9.0,
        }
    ]


def test_prompt_semantic_plan_inline_bias_targets_only_missing_families() -> None:
    model = _model(
        output_tokenizer="choice",
        semantic_plan_inline_decode_weight=3.0,
    )
    tokenizer = model.tokenizer
    modal_id = tokenizer.token_to_id["+Modal"]
    text_id = tokenizer.token_to_id["+TextContent"]
    button_id = tokenizer.token_to_id["+Button"]
    model._semantic_plan_action_scores = [
        {
            modal_id: 1.0,
            text_id: 1.0,
            button_id: 1.0,
        }
    ]
    model._semantic_plan_action_counts = [
        {
            modal_id: 1,
            text_id: 1,
            button_id: 1,
        }
    ]

    bias = model._semantic_plan_inline_bias(
        0,
        [tokenizer.bos_id, modal_id, text_id],
        (text_id, button_id),
        ("component", "component"),
    )

    assert bias is not None
    assert bias.tolist() == [0.0, 3.0]


def test_prompt_semantic_plan_bias_reserves_distinct_repeated_slots() -> None:
    from slm_training.models.choice_tokenizer import ChoiceDecodeState

    model = _model(
        output_tokenizer="choice",
        semantic_plan_decode_weight=3.0,
    )
    tokenizer = model.tokenizer
    input_id = tokenizer.token_to_id["+Input"]
    slot0 = tokenizer.token_to_id["@0"]
    close = tokenizer.token_to_id["-"]
    model._semantic_plan_action_scores = [{input_id: 1.0}]
    model._semantic_plan_action_counts = [{input_id: 2}]
    first = ChoiceDecodeState(tokenizer, slot_count=2)
    assert first.advance_id(input_id)
    assert first.advance_id(slot0)

    reserve_bias = model._semantic_plan_bias(
        0,
        (close, tokenizer.token_to_id["@1"]),
        ("structural", "slot"),
        first,
        [input_id, slot0],
    )

    assert reserve_bias is not None
    assert reserve_bias.tolist() == [3.0, 0.0]
    assert first.advance_id(close)
    second = first.clone()
    assert second.advance_id(input_id)
    assert second.advance_id(tokenizer.token_to_id["@1"])
    assert (
        model._semantic_plan_bias(
            0,
            (close,),
            ("structural",),
            second,
            [input_id, slot0, close, input_id, tokenizer.token_to_id["@1"]],
        )
        is None
    )


def test_prompt_semantic_plan_binding_bias_prefers_matching_unused_references() -> None:
    from types import SimpleNamespace

    model = _model(
        output_tokenizer="choice",
        semantic_plan_binding_decode_weight=3.0,
    )
    tokenizer = model.tokenizer
    model._semantic_plan_action_scores = [
        {
            tokenizer.token_to_id["+Input"]: 1.0,
            tokenizer.token_to_id["+Button"]: 1.0,
        }
    ]
    ref0 = tokenizer.token_to_id["&0"]
    ref1 = tokenizer.token_to_id["&1"]
    ref2 = tokenizer.token_to_id["&2"]
    close = tokenizer.token_to_id["]"]
    state = SimpleNamespace(
        mode="structural",
        frames=[SimpleNamespace(kind="variadic", expr_type="array")],
        section_types=["element:Input", "element:Slider", "element:Button"],
    )

    bias = model._semantic_plan_binding_bias(
        0,
        state,
        [tokenizer.bos_id, ref0],
        (ref0, ref1, ref2, close),
    )

    assert bias is not None
    assert bias.tolist() == [0.0, 0.0, 3.0, 0.0]


def test_prompt_semantic_plan_binding_bias_is_root_list_only() -> None:
    from types import SimpleNamespace

    model = _model(
        output_tokenizer="choice",
        semantic_plan_binding_decode_weight=3.0,
    )
    tokenizer = model.tokenizer
    model._semantic_plan_action_scores = [
        {
            tokenizer.token_to_id["+Input"]: 1.0,
        }
    ]
    nested = SimpleNamespace(
        mode="structural",
        frames=[
            SimpleNamespace(kind="component", expr_type="element:Modal"),
            SimpleNamespace(kind="variadic", expr_type="array"),
        ],
        section_types=["element:Input"],
    )

    assert (
        model._semantic_plan_binding_bias(
            0,
            nested,
            [tokenizer.bos_id],
            (tokenizer.token_to_id["&0"],),
        )
        is None
    )


def test_prompt_semantic_plan_binding_bias_reaches_stack_child_list() -> None:
    from types import SimpleNamespace

    model = _model(
        output_tokenizer="choice",
        semantic_plan_binding_decode_weight=3.0,
    )
    tokenizer = model.tokenizer
    model._semantic_plan_action_scores = [
        {
            tokenizer.token_to_id["+Input"]: 1.0,
        }
    ]
    state = SimpleNamespace(
        mode="structural",
        frames=[
            SimpleNamespace(kind="component", expr_type="element:Stack"),
            SimpleNamespace(kind="variadic", expr_type="array"),
        ],
        section_types=["element:Input", "element:Slider"],
    )

    bias = model._semantic_plan_binding_bias(
        0,
        state,
        [tokenizer.bos_id],
        (
            tokenizer.token_to_id["&0"],
            tokenizer.token_to_id["&1"],
            tokenizer.token_to_id["]"],
        ),
    )

    assert bias is not None
    assert bias.tolist() == [3.0, 0.0, 0.0]


def test_semantic_plan_role_obligations_pair_uncovered_roles() -> None:
    counts, bindings = TwoTowerModel._semantic_plan_role_obligations(
        Counter({"ImageGallery": 1}),
        {
            ":gallery.img": ("Image", "ImageGallery"),
            ":gallery.caption": ("ImageGallery", "TextContent"),
            ":gallery.hint.title": ("Callout", "TextContent"),
            ":gallery.hint.body": ("Label", "TextContent"),
            ":gallery.cta": ("Button", "FormControl"),
        },
    )

    assert counts == Counter({"TextContent": 2, "ImageGallery": 1, "Button": 1})
    assert bindings == {
        "ImageGallery": (":gallery.img", ":gallery.caption"),
        "TextContent": (":gallery.hint.title", ":gallery.hint.body"),
        "Button": (":gallery.cta",),
    }


def test_semantic_plan_role_obligations_bind_existing_action_family() -> None:
    counts, bindings = TwoTowerModel._semantic_plan_role_obligations(
        Counter({"Callout": 1, "Button": 1}),
        {
            ":dashboard.title": ("Callout", "TextContent"),
            ":dashboard.refresh": ("Button",),
        },
    )

    assert counts == Counter({"Callout": 1, "Button": 1})
    assert bindings == {
        "Callout": (":dashboard.title",),
        "Button": (":dashboard.refresh",),
    }


def test_semantic_plan_role_obligations_keep_reachable_roles_nested() -> None:
    counts, bindings = TwoTowerModel._semantic_plan_role_obligations(
        Counter({"Card": 2}),
        {
            ":dashboard.m1.value": ("TextContent",),
            ":dashboard.m2.value": ("TextContent",),
        },
        {
            ":dashboard.m1.value": ("Card", "TextContent"),
            ":dashboard.m2.value": ("Card", "TextContent"),
        },
    )

    assert counts == Counter({"Card": 2})
    assert bindings == {}


def test_semantic_plan_role_obligations_do_not_add_reachable_joint_carrier() -> None:
    counts, bindings = TwoTowerModel._semantic_plan_role_obligations(
        Counter({"Card": 5}),
        {
            ":cards.title": ("Callout", "TextContent"),
            ":cards.body": ("Callout", "TextContent"),
        },
        {
            ":cards.title": ("Card", "Callout", "TextContent"),
            ":cards.body": ("Card", "Callout", "TextContent"),
        },
    )

    assert counts == Counter({"Card": 5})
    assert bindings == {}


def test_semantic_plan_role_obligations_add_unowned_namespace_sibling() -> None:
    slots = {
        ":toolbar.text": ("TextContent",),
        **{
            f":card{index}.{role}": ("Callout", "TextContent")
            for index in range(1, 6)
            for role in ("title", "body")
        },
    }

    counts, bindings = TwoTowerModel._semantic_plan_role_obligations(
        Counter({"Card": 5}),
        slots,
        {slot: ("Card", "TextContent") for slot in slots},
    )

    assert counts == Counter({"Card": 5, "TextContent": 1})
    assert bindings == {
        "TextContent": (":toolbar.text",),
    }


def test_prompt_semantic_plan_root_bias_builds_stack_then_ends() -> None:
    from types import SimpleNamespace

    model = _model(
        output_tokenizer="choice",
        semantic_plan_root_decode_weight=2.0,
    )
    tokenizer = model.tokenizer
    model._semantic_plan_action_scores = [
        {
            tokenizer.token_to_id["+Input"]: 1.0,
            tokenizer.token_to_id["+Button"]: 1.0,
        }
    ]
    candidates = (
        tokenizer.token_to_id["+Stack"],
        tokenizer.token_to_id["+Card"],
        tokenizer.eos_id,
    )
    covered = SimpleNamespace(
        mode="structural",
        frames=[],
        section_types=["element:Input", "element:Button"],
    )
    completed = SimpleNamespace(
        mode="structural",
        frames=[],
        section_types=["element:Input", "element:Button", "element:Stack"],
    )

    build_bias = model._semantic_plan_root_bias(0, covered, None, candidates)
    end_bias = model._semantic_plan_root_bias(0, completed, None, candidates)

    assert build_bias is not None
    assert build_bias.tolist() == [2.0, 0.0, 0.0]
    assert end_bias is not None
    assert end_bias.tolist() == [0.0, 0.0, 2.0]


def test_prompt_semantic_plan_root_margin_floors_verified_target() -> None:
    from types import SimpleNamespace

    model = _model(
        output_tokenizer="choice",
        semantic_plan_root_decode_weight=8.0,
        semantic_plan_root_margin_decode_weight=2.0,
    )
    tokenizer = model.tokenizer
    model._semantic_plan_action_scores = [
        {
            tokenizer.token_to_id["+Input"]: 1.0,
            tokenizer.token_to_id["+Button"]: 1.0,
        }
    ]
    candidates = (
        tokenizer.token_to_id["+Stack"],
        tokenizer.token_to_id["+TextContent"],
        tokenizer.eos_id,
    )
    covered = SimpleNamespace(
        mode="structural",
        frames=[],
        section_types=["element:Input", "element:Button"],
    )

    bias = model._semantic_plan_root_bias(
        0,
        covered,
        None,
        candidates,
        torch.tensor([-44.0, 29.0, -10.0]),
    )

    assert bias is not None
    assert bias.tolist() == [75.0, 0.0, 0.0]


def test_semantic_plan_root_abstention_trace_is_bounded_and_deduplicated() -> None:
    from types import SimpleNamespace

    from slm_training.models.decode_stats import DecodeStats

    model = _model(output_tokenizer="choice")
    stats = DecodeStats()
    state = SimpleNamespace(mode="structural", frames=[], section_types=[])
    model._semantic_plan_root_last_abstention = {
        "reason": "verifier_rejected",
        "error_type": "ValueError",
        "error": "unreachable section",
        "planned_token_count": 20,
        "section_count": 5,
        "reference_count": 4,
    }

    model._record_semantic_plan_root_abstention(stats, row=0, position=35, state=state)
    first_evidence = dict(model._semantic_plan_root_last_abstention)
    model._semantic_plan_root_last_abstention["section_count"] = 6
    model._record_semantic_plan_root_abstention(stats, row=0, position=36, state=state)

    assert len(stats.constrained_selection_traces) == 1
    assert stats.constrained_selection_traces[0]["evidence"] == first_evidence


def test_semantic_plan_root_probe_decodes_fixed_grammar_literal() -> None:
    from types import SimpleNamespace

    model = _model(
        output_tokenizer="choice",
        semantic_plan_root_decode_weight=2.0,
    )
    tokenizer = model.tokenizer
    callout_id = tokenizer.token_to_id["+Callout"]
    model._semantic_plan_action_scores = [{callout_id: 1.0}]
    model._semantic_plan_action_counts = [{callout_id: 1}]
    model._slot_contracts = [[":status.title", ":status.body"]]
    prefix = [
        callout_id,
        tokenizer.token_to_id['#"info"'],
        tokenizer.sym_id(0),
        tokenizer.sym_id(1),
        tokenizer.token_to_id["-"],
    ]
    state = SimpleNamespace(
        mode="structural",
        frames=[],
        section_types=["element:Callout"],
    )
    candidates = (tokenizer.token_to_id["+Stack"], tokenizer.eos_id)

    bias = model._semantic_plan_root_bias(
        0,
        state,
        prefix,
        candidates,
        torch.zeros(2),
    )

    assert bias is not None
    assert bias.tolist() == [2.0, 0.0]
    assert model._semantic_plan_root_last_abstention is None


def test_prompt_semantic_plan_root_bias_waits_for_role_coverage() -> None:
    from types import SimpleNamespace

    model = _model(
        output_tokenizer="choice",
        semantic_plan_root_decode_weight=2.0,
    )
    tokenizer = model.tokenizer
    model._semantic_plan_action_scores = [
        {
            tokenizer.token_to_id["+Input"]: 1.0,
            tokenizer.token_to_id["+Button"]: 1.0,
        }
    ]
    incomplete = SimpleNamespace(
        mode="structural",
        frames=[],
        section_types=["element:Input"],
    )

    assert (
        model._semantic_plan_root_bias(
            0,
            incomplete,
            None,
            (tokenizer.token_to_id["+Stack"], tokenizer.eos_id),
        )
        is None
    )


def test_prompt_semantic_plan_root_bias_records_missing_slot_carrier_reference() -> (
    None
):
    from types import SimpleNamespace

    model = _model(
        output_tokenizer="choice",
        semantic_plan_root_decode_weight=8.0,
        semantic_plan_root_margin_decode_weight=2.0,
    )
    tokenizer = model.tokenizer
    input_id = tokenizer.token_to_id["+Input"]
    button_id = tokenizer.token_to_id["+Button"]
    text_id = tokenizer.token_to_id["+TextContent"]
    stack_id = tokenizer.token_to_id["+Stack"]
    model._semantic_plan_action_scores = [{input_id: 1.0, button_id: 1.0}]
    model._semantic_plan_action_counts = [{input_id: 1, button_id: 1}]
    model._slot_contracts = [[":auth.email", ":auth.submit", ":auth.body"]]
    model._semantic_role_candidates = [
        {
            ":auth.email": ("Input",),
            ":auth.submit": ("Button",),
            ":auth.body": ("Callout", "TextContent"),
        }
    ]
    model._semantic_plan_role_bindings = [
        {"Input": (":auth.email",), "Button": (":auth.submit",)}
    ]
    model._semantic_plan_required_root_references = [{}]
    prefix = [
        tokenizer.bos_id,
        input_id,
        tokenizer.sym_id(0),
        tokenizer.token_to_id["-"],
        button_id,
        tokenizer.sym_id(1),
        tokenizer.token_to_id["-"],
    ]
    state = SimpleNamespace(
        mode="structural",
        frames=[],
        section_types=["element:Input", "element:Button"],
    )
    candidates = (stack_id, text_id, tokenizer.eos_id)

    bias = model._semantic_plan_root_bias(
        0, state, prefix, candidates, torch.tensor([12.0, 1.0, 0.0])
    )

    assert bias is not None
    assert bias.tolist() == [0.0, 13.0, 0.0]
    assert model._semantic_plan_required_root_references == [{2: "element:TextContent"}]


def test_semantic_plan_required_reference_bias_targets_only_carrier() -> None:
    from types import SimpleNamespace

    model = _model(
        output_tokenizer="choice",
        semantic_plan_root_decode_weight=8.0,
        semantic_plan_root_margin_decode_weight=2.0,
    )
    tokenizer = model.tokenizer
    ref1 = tokenizer.token_to_id["&1"]
    ref2 = tokenizer.token_to_id["&2"]
    close = tokenizer.token_to_id["]"]
    model._semantic_plan_required_root_references = [{2: "element:TextContent"}]
    state = SimpleNamespace(
        section_types=[
            "element:Input",
            "element:Callout",
            "element:TextContent",
        ],
        frames=[
            SimpleNamespace(kind="component", expr_type="element:Stack"),
            SimpleNamespace(kind="variadic", expr_type="array", close="]"),
        ],
    )
    candidates = (ref1, ref2, close)

    bias = model._semantic_plan_required_reference_bias(
        0, state, [tokenizer.bos_id], candidates, torch.tensor([15.0, 1.0, 20.0])
    )

    assert bias is not None
    assert bias.tolist() == [0.0, 21.0, 0.0]
    assert (
        model._semantic_plan_required_reference_bias(
            0,
            state,
            [tokenizer.bos_id],
            candidates,
            torch.tensor([20.0, 1.0, 15.0]),
        )
        is None
    )
    assert (
        model._semantic_plan_required_reference_bias(
            0,
            state,
            [tokenizer.bos_id, ref2],
            candidates,
            torch.zeros(3),
        )
        is None
    )


def test_prompt_semantic_plan_root_bias_waits_for_required_family_count() -> None:
    from types import SimpleNamespace

    model = _model(
        output_tokenizer="choice",
        semantic_plan_root_decode_weight=2.0,
    )
    tokenizer = model.tokenizer
    input_id = tokenizer.token_to_id["+Input"]
    button_id = tokenizer.token_to_id["+Button"]
    model._semantic_plan_action_scores = [{input_id: 1.0, button_id: 1.0}]
    model._semantic_plan_action_counts = [{input_id: 2, button_id: 1}]
    incomplete = SimpleNamespace(
        mode="structural",
        frames=[],
        section_types=["element:Input", "element:Button"],
    )
    complete = SimpleNamespace(
        mode="structural",
        frames=[],
        section_types=["element:Input", "element:Input", "element:Button"],
    )
    candidates = (tokenizer.token_to_id["+Stack"], tokenizer.eos_id)

    assert model._semantic_plan_root_bias(0, incomplete, None, candidates) is None
    assert model._semantic_plan_root_bias(0, complete, None, candidates) is not None


def test_prompt_semantic_plan_root_bias_counts_nested_family_instances() -> None:
    from slm_training.models.choice_tokenizer import ChoiceDecodeState

    model = _model(output_tokenizer="choice", semantic_plan_root_decode_weight=2.0)
    tokenizer = model.tokenizer
    modal_id = tokenizer.token_to_id["+Modal"]
    text_id = tokenizer.token_to_id["+TextContent"]
    button_id = tokenizer.token_to_id["+Button"]
    stack_id = tokenizer.token_to_id["+Stack"]
    model._semantic_plan_action_scores = [
        {
            modal_id: 1.0,
            text_id: 1.0,
            button_id: 1.0,
        }
    ]
    model._semantic_plan_action_counts = [
        {
            modal_id: 1,
            text_id: 1,
            button_id: 1,
        }
    ]
    model._slot_contracts = [[":modal.title", ":modal.body", ":modal.confirm"]]
    prefix_tokens = [
        "<bos>",
        "+Modal",
        "@0",
        "#false",
        "[",
        "+TextContent",
        "@1",
        "-",
        "+Buttons",
        "[",
        "+Button",
        "@2",
        "-",
        "]",
        "-",
        "]",
        "-",
    ]
    prefix = [tokenizer.token_to_id[token] for token in prefix_tokens]
    state = ChoiceDecodeState(tokenizer, slot_count=3)
    for token_id in prefix[1:]:
        assert state.advance_id(token_id)

    bias = model._semantic_plan_root_bias(
        0, state, prefix, (stack_id, tokenizer.eos_id)
    )

    assert bias is not None
    assert bias.tolist() == [2.0, 0.0]


def test_prompt_semantic_plan_root_bias_follows_only_verified_closure() -> None:
    from slm_training.models.choice_tokenizer import ChoiceDecodeState

    model = _model(
        output_tokenizer="choice",
        semantic_plan_root_decode_weight=2.0,
    )
    tokenizer = model.tokenizer
    model._semantic_plan_action_scores = [
        {
            tokenizer.token_to_id["+Input"]: 1.0,
            tokenizer.token_to_id["+Button"]: 1.0,
        }
    ]
    model._slot_contracts = [[":auth.email", ":auth.submit"]]
    prefix_tokens = [
        "<bos>",
        "+Input",
        "@0",
        "-",
        "+Button",
        "@1",
        "-",
    ]
    prefix = [tokenizer.token_to_id[token] for token in prefix_tokens]
    state = ChoiceDecodeState(tokenizer, slot_count=2)
    for token_id in prefix[1:]:
        assert state.advance_id(token_id)
    candidates = (
        tokenizer.token_to_id["+Stack"],
        tokenizer.token_to_id["+Card"],
        tokenizer.eos_id,
    )

    bias = model._semantic_plan_root_bias(0, state, prefix, candidates)

    assert bias is not None
    assert bias.tolist() == [2.0, 0.0, 0.0]


def test_prompt_semantic_plan_root_bias_abstains_when_closure_is_invalid() -> None:
    from slm_training.models.choice_tokenizer import ChoiceDecodeState

    model = _model(
        output_tokenizer="choice",
        semantic_plan_root_decode_weight=2.0,
    )
    tokenizer = model.tokenizer
    model._semantic_plan_action_scores = [
        {
            tokenizer.token_to_id["+Input"]: 1.0,
        }
    ]
    model._slot_contracts = [[]]
    prefix = [
        tokenizer.bos_id,
        tokenizer.token_to_id["+Input"],
        tokenizer.token_to_id["@0"],
        tokenizer.token_to_id["-"],
    ]
    state = ChoiceDecodeState(tokenizer, slot_count=1)
    for token_id in prefix[1:]:
        assert state.advance_id(token_id)

    assert (
        model._semantic_plan_root_bias(
            0,
            state,
            prefix,
            (tokenizer.token_to_id["+Stack"], tokenizer.eos_id),
        )
        is None
    )


def test_visible_reference_bias_prefers_each_unused_bound_element_once() -> None:
    from types import SimpleNamespace

    model = _model(
        output_tokenizer="choice",
        visible_reference_decode_weight=4.0,
    )
    tokenizer = model.tokenizer
    ref0 = tokenizer.token_to_id["&0"]
    ref1 = tokenizer.token_to_id["&1"]
    close = tokenizer.token_to_id["]"]
    state = SimpleNamespace(
        current_marker="r=",
        frames=[object()],
        section_types=["element:Input", "element:Button"],
    )

    bias = model._visible_reference_completeness_bias(
        state,
        [tokenizer.bos_id, ref0],
        (ref0, ref1, close),
    )
    exhausted = model._visible_reference_completeness_bias(
        state,
        [tokenizer.bos_id, ref0, ref1],
        (ref0, ref1, close),
    )

    assert bias is not None
    assert bias.tolist() == [0.0, 4.0, 0.0]
    assert exhausted is None


def test_visible_reference_bias_reaches_structural_root_lists_only() -> None:
    from slm_training.models.choice_tokenizer import ChoiceDecodeState

    model = _model(
        output_tokenizer="choice",
        visible_reference_decode_weight=4.0,
    )
    tokenizer = model.tokenizer
    ref0 = tokenizer.token_to_id["&0"]
    close = tokenizer.token_to_id["]"]
    root = ChoiceDecodeState(tokenizer, slot_count=1)
    for token in ("+TextContent", "@0", "-", "["):
        assert root.advance_id(tokenizer.token_to_id[token])
    nested = ChoiceDecodeState(tokenizer, slot_count=1)
    for token in ("+TextContent", "@0", "-", "+Stack", "["):
        assert nested.advance_id(tokenizer.token_to_id[token])

    bias = model._visible_reference_completeness_bias(
        root,
        [tokenizer.bos_id],
        (ref0, close),
    )
    nested_bias = model._visible_reference_completeness_bias(
        nested,
        [tokenizer.bos_id],
        (ref0, close),
    )

    assert root.mode == nested.mode == "structural"
    assert bias is not None
    assert bias.tolist() == [4.0, 0.0]
    assert nested_bias is None


def test_choice_generation_evidence_preserves_reference_decisions() -> None:
    model = _model(output_tokenizer="choice")
    tokenizer = model.tokenizer
    contract = [":hero.title"]
    ids = tokenizer.encode(
        'root = Card([title])\ntitle = TextContent(":hero.title")',
        placeholders=contract,
    )
    canvas = torch.full(
        (1, len(ids) + 2),
        tokenizer.pad_id,
        dtype=torch.long,
    )
    canvas[0, : len(ids)] = torch.tensor(ids)

    evidence = model._choice_generation_evidence(canvas, [contract])

    assert evidence[0]["schema"] == "choice_decision_trace/v2"
    assert "&0" in evidence[0]["choice_tokens"]
    decisions = evidence[0]["reference_decisions"]
    assert any(row["chosen"] == "&0" for row in decisions)
    assert any("&0" in row["legal_references"] for row in decisions)
    assert any(
        row["aggregation_scope"] == "structural_nested_list" for row in decisions
    )
    assert all(row["frame_depth"] == len(row["frame_path"]) for row in decisions)


def test_choice_phase_evidence_separates_root_and_nested_lists() -> None:
    from slm_training.models.choice_tokenizer import ChoiceDecodeState

    model = _model(output_tokenizer="choice")
    tokenizer = model.tokenizer
    root = ChoiceDecodeState(tokenizer)
    assert root.advance_id(tokenizer.token_to_id["["])
    nested = ChoiceDecodeState(tokenizer)
    for token in ("+Stack", "["):
        assert nested.advance_id(tokenizer.token_to_id[token])

    root_evidence = model._choice_phase_evidence(root)
    nested_evidence = model._choice_phase_evidence(nested)

    assert root_evidence["aggregation_scope"] == "structural_root_list"
    assert root_evidence["frame_depth"] == 1
    assert nested_evidence["aggregation_scope"] == "structural_nested_list"
    assert nested_evidence["frame_depth"] == 2


def test_structural_root_reference_arity_ignores_nested_lists() -> None:
    from slm_training.models.choice_tokenizer import (
        structural_root_reference_arity,
        structural_root_reference_identity_target,
    )

    model = _model(output_tokenizer="choice")
    tokenizer = model.tokenizer
    ids = tokenizer.encode(
        "root = Stack([card, title])\n"
        "card = Card([body])\n"
        'body = TextContent(":body")\n'
        'title = TextContent(":title")',
        placeholders=[":body", ":title"],
    )

    assert structural_root_reference_arity(tokenizer, ids, slot_count=2) == 2
    references, bound = structural_root_reference_identity_target(
        tokenizer, ids, slot_count=2
    )
    assert len(references) == 2
    assert bound == 3


def test_strict_root_reference_identity_sampler_selects_only_strict_subsets() -> None:
    from slm_training.harnesses.model_build.train_loop import (
        _strict_root_reference_identity_records,
    )

    model = _model(output_tokenizer="choice")
    records = [
        ExampleRecord(
            id="strict",
            prompt="one root child",
            openui=(
                "root = Stack([card])\n"
                "card = Card([title])\n"
                'title = TextContent(":title")'
            ),
            placeholders=[":title"],
            split="train",
            source="fixture",
        ),
        ExampleRecord(
            id="all",
            prompt="all root children",
            openui=(
                "root = Stack([title, body])\n"
                'title = TextContent(":title")\n'
                'body = TextContent(":body")'
            ),
            placeholders=[":title", ":body"],
            split="train",
            source="fixture",
        ),
    ]

    selected = _strict_root_reference_identity_records(records, model.tokenizer)

    assert [record.id for record in selected] == ["strict"]


def test_rare_slot_owner_sampler_selects_records_by_label_frequency() -> None:
    from slm_training.harnesses.model_build.train_loop import (
        _rare_slot_component_owner_records,
    )

    records = [
        ExampleRecord(
            id="common",
            prompt="common",
            openui='root = TextContent(":common")',
            placeholders=[":common"],
            split="train",
            source="fixture",
        ),
        ExampleRecord(
            id="mixed",
            prompt="mixed",
            openui=(
                "root = Stack([title, field])\n"
                'title = TextContent(":title")\n'
                'field = Input(":field")'
            ),
            placeholders=[":title", ":field"],
            split="train",
            source="fixture",
        ),
    ]

    selected, counts, rare = _rare_slot_component_owner_records(
        records,
        TwoTowerModel._slot_component_owners,
        threshold=1,
    )

    assert counts == {"Input": 1, "TextContent": 2}
    assert rare == ["Input"]
    assert [record.id for record in selected] == ["mixed"]


def test_root_reference_arity_head_trains_and_biases_root_stop() -> None:
    from slm_training.models.choice_tokenizer import ChoiceDecodeState

    model = _model(
        output_tokenizer="choice",
        root_reference_arity_loss_weight=1.0,
        root_reference_arity_decode_weight=2.0,
    )
    model.train()
    assert model.root_reference_arity_head is not None
    with torch.no_grad():
        model.root_reference_arity_head.weight.zero_()
        model.root_reference_arity_head.bias.zero_()
        model.root_reference_arity_head.bias[2] = 4.0
        model.root_reference_arity_head.bias[-1] = 20.0
    record = ExampleRecord(
        id="root-reference-arity",
        prompt="stack with title and body",
        openui=(
            "root = Stack([title, body])\n"
            'title = TextContent(":title")\n'
            'body = TextContent(":body")'
        ),
        placeholders=[":title", ":body"],
        split="train",
        source="fixture",
    )
    loss = model.training_loss([record])
    loss.backward()
    auxiliary_loss = model.take_detached_auxiliary_loss()
    assert auxiliary_loss is not None
    auxiliary_loss.backward()
    assert model.root_reference_arity_head.weight.grad is not None
    assert model.root_reference_arity_head.weight.grad.abs().sum() > 0
    assert model.last_training_metrics["root_reference_arity_rows"] == 1
    assert model.last_training_metrics["root_reference_arity_accuracy"] == 0.0
    assert model.last_training_metrics["root_reference_arity_classes_mean"] == 3.0
    assert model.root_reference_arity_head.bias.grad is not None
    assert model.root_reference_arity_head.bias.grad[-1] > 0

    tokenizer = model.tokenizer
    with torch.no_grad():
        model.root_reference_arity_head.weight.zero_()
        model.root_reference_arity_head.bias.zero_()
        model.root_reference_arity_head.bias[2] = 4.0
        model.root_reference_arity_head.bias[-1] = 20.0
        model.root_reference_arity_head.bias[-1] = 20.0
    state = ChoiceDecodeState(tokenizer, slot_count=2)
    for token in ("+TextContent", "@0", "-", "+TextContent", "@1", "-", "["):
        assert state.advance_id(tokenizer.token_to_id[token])
    ctx, ctx_pad = model._encode_context(["stack with title and body"])
    candidates = (tokenizer.token_to_id["&0"], tokenizer.token_to_id["]"])
    continue_bias = model._root_reference_arity_bias(ctx, ctx_pad, state, candidates)
    assert continue_bias is not None and continue_bias[0] > continue_bias[1]
    assert state.advance_id(tokenizer.token_to_id["&0"])
    assert state.advance_id(tokenizer.token_to_id["&1"])
    stop_bias = model._root_reference_arity_bias(ctx, ctx_pad, state, candidates)
    assert stop_bias is not None and stop_bias[1] > stop_bias[0]


def test_root_reference_identity_head_trains_and_prefers_uncovered_identity() -> None:
    from slm_training.models.choice_tokenizer import ChoiceDecodeState

    model = _model(
        output_tokenizer="choice",
        root_reference_identity_loss_weight=1.0,
        root_reference_identity_negative_weight=4.0,
        root_reference_identity_decode_weight=2.0,
    )
    model.train()
    assert model.root_reference_identity_head is not None
    with torch.no_grad():
        model.root_reference_identity_head.weight.zero_()
        model.root_reference_identity_head.bias.fill_(-4.0)
        model.root_reference_identity_head.bias[-1] = 20.0
    record = ExampleRecord(
        id="root-reference-identity",
        prompt="stack with title and body",
        openui=(
            "root = Stack([card])\n"
            "card = Card([title, body])\n"
            'title = TextContent(":title")\n'
            'body = TextContent(":body")'
        ),
        placeholders=[":title", ":body"],
        split="train",
        source="fixture",
    )
    loss = model.training_loss([record])
    loss.backward()
    auxiliary_loss = model.take_detached_auxiliary_loss()
    assert auxiliary_loss is not None
    auxiliary_loss.backward()
    assert model.root_reference_identity_head.weight.grad is not None
    assert model.root_reference_identity_head.weight.grad.abs().sum() > 0
    assert model.last_training_metrics["root_reference_identity_rows"] == 1
    assert model.last_training_metrics["root_reference_identity_exact_accuracy"] == 0
    assert model.last_training_metrics["root_reference_identity_negative_accuracy"] == 1
    assert model.last_training_metrics["root_reference_identity_negative_rows"] == 1
    assert model.last_training_metrics["root_reference_identity_classes_mean"] == 3

    tokenizer = model.tokenizer
    with torch.no_grad():
        model.root_reference_identity_head.weight.zero_()
        model.root_reference_identity_head.bias.fill_(-4.0)
        model.root_reference_identity_head.bias[1] = 4.0
    state = ChoiceDecodeState(tokenizer, slot_count=2)
    for token in ("+TextContent", "@0", "-", "+TextContent", "@1", "-", "["):
        assert state.advance_id(tokenizer.token_to_id[token])
    ctx, ctx_pad = model._encode_context(["stack with title and body"])
    candidates = (
        tokenizer.token_to_id["&0"],
        tokenizer.token_to_id["&1"],
        tokenizer.token_to_id["]"],
    )
    scores = torch.tensor([2.0, 3.0, 4.0])
    first = model._root_reference_identity_bias(
        ctx, ctx_pad, state, [], candidates, scores
    )
    assert first is not None
    adjusted_first = scores + first
    assert adjusted_first[1] > adjusted_first[0]
    assert adjusted_first[:2].max() == scores[:2].max()
    assert adjusted_first[2] == scores[2]
    prefix = [tokenizer.token_to_id["&1"]]
    second = model._root_reference_identity_bias(
        ctx, ctx_pad, state, prefix, candidates, scores
    )
    assert second is not None
    adjusted_second = scores + second
    assert adjusted_second[0] > adjusted_second[1]
    assert adjusted_second[:2].max() == scores[:2].max()
    assert adjusted_second[2] == scores[2]


def test_projection_with_features_accepts_sliced_hidden() -> None:
    """Compiler/tree scorers project [D] and [N,D] slices; with one request's
    runtime symbol features active the projection must match the [B,T,D]
    path instead of crashing on the feature einsum."""
    denoiser = DenoiserTower(vocab_size=17, d_model=8, n_layers=1, n_heads=2)
    features = torch.zeros((1, 17, 8))
    features[:, 3, :] = 0.5
    denoiser.set_runtime_symbol_features(features)
    hidden3 = torch.randn(1, 4, 8)
    full = denoiser.project(hidden3)
    single = denoiser.project(hidden3[0, 2])
    assert torch.allclose(single, full[0, 2], atol=1e-5)
    candidates = torch.tensor([3, 5])
    gathered = denoiser.project(hidden3[0, 2], candidates)
    assert torch.allclose(gathered, full[0, 2].index_select(-1, candidates), atol=1e-5)
    # Multi-request features cannot be attributed to a sliced row: fail closed.
    denoiser.set_runtime_symbol_features(torch.zeros((2, 17, 8)))
    with pytest.raises(ValueError, match="B,T,D"):
        denoiser.project(hidden3[0, 2])


def test_completion_forest_uses_active_binder_and_symbol_spaces(monkeypatch) -> None:
    from slm_training.dsl.grammar.fastpath import compiler_draft

    tokenizer = DSLNativeTokenizer.build()
    start = build_completion_forest(tokenizer, [])
    assert start.coverage == "complete"
    assert tokenizer.bind_id(0) in start.candidate_ids
    assert tokenizer.bind_id(1) not in start.candidate_ids
    bos_start = build_completion_forest(tokenizer, [tokenizer.bos_id])
    assert tokenizer.bind_id(0) in bos_start.candidate_ids
    assert tokenizer.bind_id(1) not in bos_start.candidate_ids
    assert tokenizer.eos_id not in bos_start.candidate_ids
    after_newline = build_completion_forest(
        tokenizer, [tokenizer.bos_id, tokenizer.token_to_id["NL"]]
    )
    assert tokenizer.token_to_id["NL"] not in after_newline.candidate_ids
    assert tokenizer.bind_id(0) in after_newline.candidate_ids
    root_value = build_completion_forest(
        tokenizer,
        [
            tokenizer.bos_id,
            tokenizer.bind_id(0),
            tokenizer.token_to_id["="],
        ],
        slot_contract=[":hero.title"],
    )
    assert root_value.candidate_ids
    assert all(
        compiler_draft._semantic_kind(tokenizer, token_id) == "component"
        for token_id in root_value.candidate_ids
    )
    child_value = build_completion_forest(
        tokenizer,
        [
            tokenizer.bos_id,
            *tokenizer.encode("root=Stack([b1])", add_special=False),
            tokenizer.token_to_id["NL"],
            tokenizer.bind_id(1),
            tokenizer.token_to_id["="],
        ],
    )
    assert child_value.candidate_ids
    assert all(
        compiler_draft._semantic_kind(tokenizer, token_id) == "component"
        for token_id in child_value.candidate_ids
    )
    children = build_completion_forest(
        tokenizer,
        [
            tokenizer.bos_id,
            *tokenizer.encode("root=Stack([", add_special=False),
        ],
        slot_contract=[":hero.title"],
    )
    assert not (set(children.candidate_ids) & set(tokenizer.kind_ids("sym")))
    assert not (set(children.candidate_ids) & set(tokenizer.kind_ids("lit")))
    assert tokenizer.bind_id(1) in children.candidate_ids
    assert tokenizer.token_to_id["Stack"] in children.candidate_ids

    nested_children = build_completion_forest(
        tokenizer,
        [
            tokenizer.bos_id,
            *tokenizer.encode("root=Stack([Stack([", add_special=False),
        ],
    )
    assert not (set(nested_children.candidate_ids) & set(tokenizer.kind_ids("lit")))

    typed_string = build_completion_forest(
        tokenizer,
        tokenizer.encode("root=TextArea(", add_special=False),
        slot_contract=[":hero.title"],
    )
    assert tokenizer.sym_id(0) in typed_string.candidate_ids
    for token in ("true", "false", "null", "LIT_NUM", "LIT_END"):
        assert tokenizer.token_to_id[token] not in typed_string.candidate_ids

    monkeypatch.setattr(
        compiler_draft,
        "_official_schema",
        lambda: {
            "properties": {"TextContent": {}},
            "$defs": {"TextContent": {"properties": {"text": {"type": "string"}}}},
        },
    )
    prefix = tokenizer.encode("root=TextContent(", add_special=False)
    forest = build_completion_forest(tokenizer, prefix, slot_contract=[":hero.title"])
    assert forest.coverage == "complete"
    assert tokenizer.sym_id(0) in forest.candidate_ids
    assert tokenizer.sym_id(1) not in forest.candidate_ids
    assert set(forest.candidate_ids) == {tokenizer.sym_id(0)}
    # Every emitted edge is accepted by the grammar and has a reachable
    # continuation; candidate policy must come from parser state, not from
    # a list of forbidden punctuation or component names.
    assert forest.paths
    assert all(path.token_ids for path in forest.paths)

    complete = build_completion_forest(
        tokenizer,
        [
            tokenizer.bos_id,
            *tokenizer.encode('root=TextContent(":hero.title")', add_special=False),
        ],
        slot_contract=[":hero.title"],
    )
    continuation_ids = set(
        compiler_draft.allowed_id_set(
            tokenizer,
            frozenset({"$END", "_NL", "NAME", "STATE_NAME", "COMMENT", "WS_INLINE"}),
        )
        or set()
    ) | {tokenizer.eos_id}
    assert set(complete.candidate_ids) <= continuation_ids
    assert tokenizer.eos_id in complete.candidate_ids


def test_min_content_contract_withholds_eos_until_components_emitted() -> None:
    # A4: an empty/underfull but grammatically complete layout must not be a
    # legal completion while the grammar still offers a way to add content.
    from slm_training.dsl.grammar.fastpath.compiler_draft import (
        emitted_component_count,
    )

    tokenizer = DSLNativeTokenizer.build()
    contract = [":hero.title"]
    prefix = [
        tokenizer.bos_id,
        *tokenizer.encode('root=TextContent(":hero.title")', add_special=False),
    ]
    assert emitted_component_count(tokenizer, prefix) == 1

    # Contract off (default): EOS admitted for the complete document.
    off = build_completion_forest(tokenizer, prefix, slot_contract=contract)
    assert tokenizer.eos_id in off.candidate_ids

    # Floor met (1 component >= 1): EOS still admitted.
    met = build_completion_forest(
        tokenizer, prefix, slot_contract=contract, min_content=1
    )
    assert tokenizer.eos_id in met.candidate_ids

    # Floor unmet (1 component < 2): EOS withheld, but a non-EOS continuation
    # remains so decode is forced to add content rather than dead-end.
    unmet = build_completion_forest(
        tokenizer, prefix, slot_contract=contract, min_content=2
    )
    assert tokenizer.eos_id not in unmet.candidate_ids
    assert unmet.paths
    assert any(path.token_ids for path in unmet.paths)


def test_effective_min_content_auto_from_inventory() -> None:
    # decode_min_content == -1 derives the floor from distinct slot roots.
    model = TwoTowerModel.from_records(
        [
            canonicalize_example_template_markers(
                ExampleRecord(
                    id="a",
                    prompt="Hero",
                    openui='root = Stack([t])\nt = TextContent(":hero.title")',
                    placeholders=[":hero.title"],
                )
            )
        ],
        config=TwoTowerConfig(
            d_model=32,
            n_heads=4,
            context_layers=1,
            denoiser_layers=1,
            seed=0,
            decode_min_content=-1,
        ),
        device="cpu",
    )
    assert model._effective_min_content([":hero.title", ":hero.body"]) == 1
    assert model._effective_min_content([":a.x", ":b.y"]) == 2
    assert model._effective_min_content(None) == 0
    model.config.decode_min_content = 3
    assert model._effective_min_content(None) == 3
    model.config.decode_min_content = 0
    assert model._effective_min_content([":a.x", ":b.y"]) == 0


def test_complete_ast_uses_lark_when_official_parser_is_unavailable(
    monkeypatch,
) -> None:
    from slm_training.dsl import lang_core
    from slm_training.dsl.grammar.fastpath import compiler_draft

    compiler_draft._generated_ast_is_complete.cache_clear()
    monkeypatch.setattr(
        lang_core,
        "parse",
        lambda _source: (_ for _ in ()).throw(RuntimeError("offline")),
    )

    assert compiler_draft._generated_ast_is_complete(
        'root = TextContent(":hero.title")'
    )
    assert not compiler_draft._generated_ast_is_complete("root = TextContent(")


def test_completion_forest_uses_schema_property_order_for_enums(monkeypatch) -> None:
    from slm_training.dsl.grammar.fastpath import compiler_draft

    tokenizer = DSLNativeTokenizer.build()
    schema = {
        "properties": {"Stack": {}},
        "$defs": {
            "Stack": {
                "properties": {
                    "children": {"type": "array"},
                    "direction": {"enum": ["row", "column"]},
                }
            }
        },
    }
    monkeypatch.setattr(compiler_draft, "_official_schema", lambda: schema)
    prefix = tokenizer.encode("root=Stack([],", add_special=False)
    forest = build_completion_forest(tokenizer, prefix)
    assert forest.coverage == "complete"
    assert set(forest.candidate_ids) == {
        tokenizer.token_to_id["STR:row"],
        tokenizer.token_to_id["STR:column"],
    }
    completed_enum = build_completion_forest(
        tokenizer,
        tokenizer.encode('root=Stack([],"column"', add_special=False),
    )
    assert set(completed_enum.candidate_ids) == {tokenizer.token_to_id[")"]}


def test_completion_forest_enforces_generated_schema_arity(monkeypatch) -> None:
    from slm_training.dsl.grammar.fastpath import compiler_draft

    tokenizer = DSLNativeTokenizer.build()
    monkeypatch.setattr(
        compiler_draft,
        "_official_schema",
        lambda: {
            "properties": {"SelectItem": {}},
            "$defs": {
                "SelectItem": {
                    "properties": {
                        "value": {"type": "string"},
                        "label": {"type": "string"},
                    },
                    "required": ["value", "label"],
                }
            },
        },
    )

    first = build_completion_forest(
        tokenizer, tokenizer.encode('root=SelectItem(":value"', add_special=False)
    )
    assert set(first.candidate_ids) == {tokenizer.token_to_id[","]}

    complete = build_completion_forest(
        tokenizer,
        tokenizer.encode('root=SelectItem(":value",":label"', add_special=False),
    )
    assert set(complete.candidate_ids) == {tokenizer.token_to_id[")"]}


def test_completion_forest_closes_max_arity_after_array_argument() -> None:
    tokenizer = DSLNativeTokenizer.build()
    prefix = tokenizer.encode(
        'root=TabItem(":value",":label",[TextContent(":content")]',
        add_special=False,
    )

    forest = build_completion_forest(
        tokenizer,
        prefix,
        slot_contract=[":value", ":label", ":content"],
        enforce_schema_component_types=True,
    )

    assert set(forest.candidate_ids) == {tokenizer.token_to_id[")"]}


def test_completion_forest_has_no_lexer_string_literal_frame() -> None:
    from slm_training.models.dsl_tokenizer import TokenKind

    tokenizer = DSLNativeTokenizer.build()
    contract = [":value", ":label"]
    prefix = tokenizer.encode('root=RadioItem(":value",":label",', add_special=False)
    forest = build_completion_forest(tokenizer, prefix, slot_contract=contract)

    assert "LIT_STR" not in tokenizer.token_to_id
    assert not (set(forest.candidate_ids) & tokenizer.kind_ids(TokenKind.BYTE))


def test_completion_forest_keeps_numeric_literal_inside_lexer_frame() -> None:
    from slm_training.dsl.grammar.fastpath.engine import OpenUIIncrementalEngine
    from slm_training.models.grammar import GrammarDecodeState
    from slm_training.models.dsl_tokenizer import TokenKind

    tokenizer = DSLNativeTokenizer.build()
    number_slot = tokenizer.encode(
        'root=Slider(":value","discrete",', add_special=False
    )
    opener = tokenizer.token_to_id["LIT_NUM"]
    opening = build_completion_forest(tokenizer, number_slot, slot_contract=[":value"])
    opener_path = next(path for path in opening.paths if path.token_ids[0] == opener)
    assert opener_path.token_ids == (opener,)

    prefix = [*number_slot, opener]

    empty = build_completion_forest(tokenizer, prefix, slot_contract=[":value"])
    byte_ids = tokenizer.kind_ids(TokenKind.BYTE)
    assert set(empty.candidate_ids)
    assert set(empty.candidate_ids) <= byte_ids
    assert tokenizer.token_to_id["B:30"] in empty.candidate_ids
    assert tokenizer.token_to_id["B:61"] not in empty.candidate_ids
    assert tokenizer.token_to_id["LIT_END"] not in empty.candidate_ids
    digit_path = next(
        path
        for path in empty.paths
        if path.token_ids[0] == tokenizer.token_to_id["B:30"]
    )
    assert digit_path.token_ids == (tokenizer.token_to_id["B:30"],)

    with_digit = build_completion_forest(
        tokenizer,
        [*prefix, tokenizer.token_to_id["B:30"]],
        slot_contract=[":value"],
    )
    assert tokenizer.token_to_id["LIT_END"] in with_digit.candidate_ids
    close_path = next(
        path
        for path in with_digit.paths
        if path.token_ids[0] == tokenizer.token_to_id["LIT_END"]
    )
    assert close_path.token_ids == (tokenizer.token_to_id["LIT_END"],)

    complete_number = tokenizer.encode(
        'root=Slider(":value","discrete",1e1', add_special=False
    )
    state = GrammarDecodeState(engine=OpenUIIncrementalEngine())
    for token_id in complete_number:
        state.advance_token(tokenizer, token_id)
    assert state.prefix_text.replace(" ", "") == tokenizer.decode(
        complete_number, preserve_trailing_newline=True
    ).replace(" ", "")
    after_number = build_completion_forest(
        tokenizer,
        complete_number,
        state=state,
        slot_contract=[":value"],
    )
    assert set(after_number.candidate_ids) == {tokenizer.token_to_id[","]}


def test_completion_forest_enforces_primitive_array_item_schema() -> None:
    tokenizer = DSLNativeTokenizer.build()
    contract = [":value"]
    prefix = tokenizer.encode(
        'root=Slider(":value","continuous",0,100,1,[', add_special=False
    )
    item = build_completion_forest(tokenizer, prefix, slot_contract=contract)
    assert tokenizer.token_to_id["LIT_NUM"] in item.candidate_ids
    assert tokenizer.token_to_id["SwitchGroup"] not in item.candidate_ids

    completed = build_completion_forest(
        tokenizer,
        tokenizer.encode(
            'root=Slider(":value","continuous",0,100,1,[40', add_special=False
        ),
        slot_contract=contract,
    )
    assert set(completed.candidate_ids) == {
        tokenizer.token_to_id[","],
        tokenizer.token_to_id["]"],
    }


def test_completion_forest_rejects_enum_absent_from_fixed_schema_vocabulary(
    monkeypatch,
) -> None:
    from slm_training.dsl.grammar.fastpath import compiler_draft

    tokenizer = DSLNativeTokenizer.build()
    schema = {
        "$defs": {
            "Stack": {
                "properties": {
                    "children": {"type": "array"},
                    "align": {"type": "string", "enum": ["schema-only"]},
                },
                "required": ["children"],
            }
        },
        "properties": {"Stack": {}},
    }
    monkeypatch.setattr(compiler_draft, "_official_schema", lambda: schema)
    prefix = [
        tokenizer.bos_id,
        *tokenizer.encode("root=Stack([] ,", add_special=False),
    ]
    with pytest.raises(ValueError, match="free-form output string is forbidden"):
        build_completion_forest(tokenizer, prefix)


def test_active_call_ignores_nested_array_and_call_commas() -> None:
    from slm_training.dsl.grammar.fastpath import compiler_draft
    from slm_training.dsl.grammar.fastpath.engine import OpenUIIncrementalEngine

    engine = OpenUIIncrementalEngine()
    assert engine.set_prefix('root=Stack([TextContent(":a"),TextContent(":b")')
    assert compiler_draft._active_call(engine) == ("Stack", 0, 1)

    assert engine.set_prefix(
        'root=Form("name",Buttons([]),[FormControl(":label",":input")'
    )
    assert compiler_draft._active_call(engine) == ("Form", 2, 3)


def test_completion_forest_tracks_forward_binder_scope() -> None:
    tokenizer = DSLNativeTokenizer.build()
    first = [
        tokenizer.bos_id,
        *tokenizer.encode("root=Stack([", add_special=False),
    ]
    forest = build_completion_forest(tokenizer, first)
    assert tokenizer.bind_id(1) in forest.candidate_ids
    assert tokenizer.bind_id(2) not in forest.candidate_ids
    assert tokenizer.bind_id(0) not in forest.candidate_ids

    second = [*first, tokenizer.bind_id(1), tokenizer.token_to_id[","]]
    forest = build_completion_forest(tokenizer, second)
    assert tokenizer.bind_id(1) not in forest.candidate_ids
    assert tokenizer.bind_id(2) in forest.candidate_ids

    nested = [
        tokenizer.bos_id,
        *tokenizer.encode("root=Stack([Stack([b1]),", add_special=False),
    ]
    forest = build_completion_forest(tokenizer, nested)
    assert tokenizer.bind_id(1) in forest.candidate_ids

    declaration = [
        tokenizer.bos_id,
        *tokenizer.encode("root=Stack([b1,b2])", add_special=False),
        tokenizer.token_to_id["NL"],
    ]
    forest = build_completion_forest(tokenizer, declaration)
    assert set(forest.candidate_ids) & set(tokenizer.kind_ids("bind")) == {
        tokenizer.bind_id(1)
    }

    unresolved = [
        tokenizer.bos_id,
        *tokenizer.encode('root=Stack([b1],"column")', add_special=False),
    ]
    forest = build_completion_forest(tokenizer, unresolved)
    assert tokenizer.eos_id not in forest.candidate_ids
    postfix_ids = {
        tokenizer.token_to_id[token]
        for token in ("[", ".", "?", "+", "-", "*", "/", "%", ">", "<")
    }
    assert not (set(forest.candidate_ids) & postfix_ids)

    resolved = [
        tokenizer.bos_id,
        *tokenizer.encode(
            'root=Stack([b1],"column")\nb1=TextContent(":hero.title")',
            add_special=False,
        ),
    ]
    forest = build_completion_forest(tokenizer, resolved, slot_contract=[":hero.title"])
    assert tokenizer.eos_id in forest.candidate_ids


def test_completion_forest_propagates_typed_array_use_to_forward_declaration() -> None:
    tokenizer = DSLNativeTokenizer.build()
    prefix = tokenizer.encode(
        'root=Tabs([b1])\nb1=',
        add_special=True,
    )[:-1]

    control = build_completion_forest(tokenizer, prefix)
    assert tokenizer.token_to_id["Separator"] in control.candidate_ids

    forest = build_completion_forest(
        tokenizer,
        prefix,
        enforce_schema_component_types=True,
        explain=True,
    )

    assert tokenizer.token_to_id["TabItem"] in forest.candidate_ids
    assert tokenizer.token_to_id["Separator"] not in forest.candidate_ids
    assert any(
        evidence.stage is ConstraintStage.SCHEMA
        and evidence.reason_code == "schema_forward_binder_component_type"
        and evidence.candidate_id == tokenizer.token_to_id["Separator"]
        for evidence in forest.evidence
    )


def test_forward_declaration_without_typed_use_keeps_component_choice_open() -> None:
    tokenizer = DSLNativeTokenizer.build()
    prefix = tokenizer.encode(
        'root=Stack([b1])\nb1=',
        add_special=True,
    )[:-1]

    forest = build_completion_forest(tokenizer, prefix)

    assert tokenizer.token_to_id["TextContent"] in forest.candidate_ids
    assert tokenizer.token_to_id["Separator"] in forest.candidate_ids


@pytest.mark.parametrize(
    ("prefix_text", "allowed", "rejected"),
    [
        ('root=Form(":name",', "Buttons", "Button"),
        (
            'root=Form(":name",Buttons([]),[FormControl(":label",',
            "Input",
            "TextContent",
        ),
    ],
)
def test_completion_forest_enforces_direct_component_property_schema(
    prefix_text: str,
    allowed: str,
    rejected: str,
) -> None:
    tokenizer = DSLNativeTokenizer.build()
    prefix = tokenizer.encode(prefix_text, add_special=True)[:-1]

    control = build_completion_forest(tokenizer, prefix)
    assert tokenizer.token_to_id[rejected] in control.candidate_ids

    forest = build_completion_forest(
        tokenizer,
        prefix,
        enforce_schema_component_types=True,
        explain=True,
    )
    assert tokenizer.token_to_id[allowed] in forest.candidate_ids
    assert tokenizer.token_to_id[rejected] not in forest.candidate_ids
    assert tokenizer.token_to_id["("] not in forest.candidate_ids
    assert any(
        evidence.stage is ConstraintStage.SCHEMA
        and evidence.reason_code == "schema_slot_component_type"
        and evidence.candidate_id == tokenizer.token_to_id[rejected]
        for evidence in forest.evidence
    )


def test_completion_forest_propagates_direct_component_binder_type() -> None:
    tokenizer = DSLNativeTokenizer.build()
    prefix = tokenizer.encode(
        'root=Form(":name",b1,[FormControl(":label",Input("email"))])\nb1=',
        add_special=True,
    )[:-1]

    forest = build_completion_forest(
        tokenizer,
        prefix,
        enforce_schema_component_types=True,
    )

    assert tokenizer.token_to_id["Buttons"] in forest.candidate_ids
    assert tokenizer.token_to_id["Button"] not in forest.candidate_ids


def test_gold_decisions_follow_compiler_forest() -> None:
    tokenizer = DSLNativeTokenizer.build()
    target = tokenizer.encode(
        'root=Card([title,body])\ntitle=TextContent(":hero.title")\n'
        'body=TextContent(":hero.body")',
        add_special=True,
    )
    positions = gold_compiler_decision_positions(
        tokenizer, target, slot_contract=[":hero.title", ":hero.body"]
    )
    selected = {tokenizer.id_to_token[target[position]] for position in positions}
    assert {"<BIND_0>", "Card", "<BIND_1>", "TextContent"} <= selected
    kinds = {decision.kind for decision in gold_compiler_decisions(tokenizer, target)}
    assert {
        "bind_declaration_root",
        "bind_reference_root_children",
        "component_root",
        "component_bound",
    } <= kinds
    assert "grammar_rsqb_root_populated" in kinds
    assert "grammar_comma" in kinds
    decisions = gold_compiler_decisions(tokenizer, target)
    assert all(len(decision.candidate_ids) > 1 for decision in decisions)
    assert all(
        target[decision.position] in decision.candidate_ids for decision in decisions
    )
    assert all(
        decision.is_semantic_role
        for decision in decisions
        if decision.kind.startswith(("component_", "bind_"))
    )
    assert all(
        not decision.is_semantic_role
        for decision in decisions
        if decision.kind.startswith("grammar_")
    )
    empty = tokenizer.encode("root=Stack([])", add_special=True)
    assert "grammar_rsqb_root_empty" not in {
        decision.kind for decision in gold_compiler_decisions(tokenizer, empty)
    }
    bound_empty = tokenizer.encode(
        "root=Stack([child])\nchild=Stack([])", add_special=True
    )
    assert "grammar_rsqb_bound_empty" not in {
        decision.kind for decision in gold_compiler_decisions(tokenizer, bound_empty)
    }


def test_grammar_state_advances_lexer_literals_as_source() -> None:
    tokenizer = DSLNativeTokenizer.build()
    state = make_grammar_state()
    prefix = [
        tokenizer.bos_id,
        *tokenizer.encode('root=Stack([Form("m"', add_special=False),
    ]
    for token_id in prefix[1:]:
        state.advance_token(tokenizer, token_id)
    forest = build_completion_forest(tokenizer, prefix, state=state)
    assert state.prefix_text.endswith('Form("m"')
    assert set(forest.candidate_ids) == {tokenizer.token_to_id[","]}


def test_compiler_alignment_loss_trains_gold_semantic_states() -> None:
    model = _model()
    model.config.compiler_alignment_loss_weight = 1.0
    record = ExampleRecord(
        id="alignment",
        prompt="card",
        openui='root = Card([title])\ntitle = TextContent(":hero.title")',
        placeholders=[":hero.title"],
        split="train",
        source="fixture",
    )
    loss = model.training_loss([record])
    assert torch.isfinite(loss)
    assert model.last_training_metrics["compiler_alignment_rows"] == 1
    assert model.last_training_metrics["compiler_alignment_loss"] > 0.0
    assert model.last_training_metrics["compiler_alignment_candidate_count_mean"] > 1


def test_compiler_alignment_margin_reports_legal_branch_violations() -> None:
    model = _model()
    model.config.compiler_alignment_loss_weight = 1.0
    model.config.compiler_alignment_margin = 100.0
    record = ExampleRecord(
        id="alignment-margin",
        prompt="card",
        openui='root = Card([title])\ntitle = TextContent(":hero.title")',
        placeholders=[":hero.title"],
        split="train",
        source="fixture",
    )
    loss = model.training_loss([record])
    assert torch.isfinite(loss)
    assert model.last_training_metrics["compiler_alignment_margin_loss"] > 0.0
    assert (
        model.last_training_metrics["compiler_alignment_margin_violation_rate"] == 1.0
    )


def test_compiler_alignment_can_stratify_grammar_decision_kinds() -> None:
    model = _model()
    model.config.compiler_alignment_loss_weight = 1.0
    model.config.compiler_alignment_stratified = True
    record = ExampleRecord(
        id="alignment-stratified",
        prompt="card",
        openui=(
            "root = Card([title, body])\n"
            'title = TextContent(":hero.title")\n'
            'body = TextContent(":hero.body")'
        ),
        placeholders=[":hero.title", ":hero.body"],
        split="train",
        source="fixture",
    )
    loss = model.training_loss([record])
    assert torch.isfinite(loss)
    metrics = model.last_training_metrics
    assert metrics["compiler_alignment_rows"] > 1
    assert metrics["compiler_alignment_bind_declaration_root_rows"] == 1
    assert "compiler_alignment_bind_declaration_bound_rows" not in metrics
    assert metrics["compiler_alignment_bind_reference_root_children_rows"] == 1
    assert metrics["compiler_alignment_component_root_rows"] == 1
    assert metrics["compiler_alignment_component_bound_rows"] == 1
    assert metrics["compiler_alignment_grammar_rsqb_root_populated_rows"] == 1
    assert metrics["compiler_alignment_grammar_comma_rows"] == 1
    assert metrics["compiler_alignment_grammar_rsqb_root_populated_loss"] >= 0.0


def test_component_inventory_supervision_trains_prompt_level_component_set() -> None:
    model = _model(component_inventory_loss_weight=1.0)
    model.train()
    record = ExampleRecord(
        id="inventory",
        prompt="card with a title",
        openui='root = Card([title])\ntitle = TextContent(":hero.title")',
        placeholders=[":hero.title"],
        split="train",
        source="fixture",
    )

    loss = model.training_loss([record])
    loss.backward()

    assert torch.isfinite(loss)
    assert model.component_inventory_head is not None
    assert model.component_inventory_head.weight.grad is not None
    assert model.component_inventory_head.weight.grad.abs().sum() > 0
    metrics = model.last_training_metrics
    assert metrics["component_inventory_loss"] > 0.0
    assert 0.0 <= metrics["component_inventory_topk_recall"] <= 1.0
    assert metrics["component_inventory_positive_count_mean"] == 2.0


def test_component_inventory_bias_only_scores_compiler_legal_components() -> None:
    model = _model(
        component_inventory_loss_weight=1.0,
        component_inventory_decode_weight=2.0,
    )
    assert model.component_inventory_head is not None
    tokenizer = model.tokenizer
    card = tokenizer.token_to_id["Card"]
    stack = tokenizer.token_to_id["Stack"]
    lparen = tokenizer.token_to_id["("]
    with torch.no_grad():
        model.component_inventory_head.weight.zero_()
        model.component_inventory_head.bias.zero_()
        model.component_inventory_head.bias[card] = 3.0
        model.component_inventory_head.bias[stack] = -1.0
    ctx, ctx_pad = model._encode_context(["card"])

    bias = model._component_inventory_bias(ctx, ctx_pad, (stack, lparen, card))

    assert bias is not None
    assert bias.tolist() == [-2.0, 0.0, 6.0]


def test_component_plan_supervises_root_role_and_bound_counts() -> None:
    model = _model(component_plan_loss_weight=1.0)
    model.train()
    record = ExampleRecord(
        id="component-plan",
        prompt="card with two labels",
        openui=(
            "root = Card([title, body])\n"
            'title = TextContent(":hero.title")\n'
            'body = TextContent(":hero.body")'
        ),
        placeholders=[":hero.title", ":hero.body"],
        split="train",
        source="fixture",
    )

    loss = model.training_loss([record])
    loss.backward()

    assert torch.isfinite(loss)
    assert model.component_plan_head is not None
    assert model.component_plan_head.weight.grad is not None
    metrics = model.last_training_metrics
    assert metrics["component_plan_loss"] > 0.0
    assert metrics["component_plan_bound_loss"] >= 0.0
    assert 0.0 <= metrics["component_plan_root_accuracy"] <= 1.0
    assert 0.0 <= metrics["component_plan_bound_topk_recall"] <= 1.0
    assert metrics["component_plan_bound_count_mae"] >= 0.0


def test_component_plan_bias_is_role_conditioned_and_count_aware() -> None:
    model = _model(
        component_plan_loss_weight=1.0,
        component_plan_decode_weight=2.0,
    )
    assert model.component_plan_head is not None
    tokenizer = model.tokenizer
    card = tokenizer.token_to_id["Card"]
    text = tokenizer.token_to_id["TextContent"]
    stack = tokenizer.token_to_id["Stack"]
    with torch.no_grad():
        model.component_plan_head.weight.zero_()
        model.component_plan_head.bias.zero_()
        vocab = tokenizer.vocab_size
        model.component_plan_head.bias[card] = 3.0
        model.component_plan_head.bias[vocab + text] = 4.0
    ctx, ctx_pad = model._encode_context(["card with labels"])

    root_bias = model._component_plan_bias(
        ctx,
        ctx_pad,
        [],
        (stack, card),
        ("component_root", "component_root"),
    )
    bound_bias_before = model._component_plan_bias(
        ctx,
        ctx_pad,
        [card],
        (card, text),
        ("component_bound", "component_bound"),
    )
    bound_bias_after = model._component_plan_bias(
        ctx,
        ctx_pad,
        [card, text],
        (card, text),
        ("component_bound", "component_bound"),
    )

    assert root_bias is not None and root_bias.tolist() == [0.0, 6.0]
    assert bound_bias_before is not None
    assert bound_bias_after is not None
    assert bound_bias_before[1] > bound_bias_before[0]
    assert bound_bias_after[1] < bound_bias_before[1]


def test_component_edges_come_from_ast_and_partial_reference_graph() -> None:
    tokenizer = DSLNativeTokenizer.build()
    root = {
        "type": "element",
        "typeName": "Card",
        "props": {
            "children": [
                {
                    "type": "element",
                    "typeName": "TextContent",
                    "props": {"text": ":title"},
                }
            ]
        },
    }
    card = tokenizer.token_to_id["Card"]
    text = tokenizer.token_to_id["TextContent"]
    assert semantic_component_edges(root, tokenizer) == ((card, text),)
    prefix = tokenizer.encode("root = Card([title])\ntitle =", add_special=False)
    assert active_parent_component_ids(tokenizer, prefix) == (card,)
    assert active_parent_component_ids(tokenizer, [tokenizer.bos_id, *prefix]) == (
        card,
    )
    assert active_declaration_binder_id(tokenizer, prefix) == tokenizer.bind_id(1)


def test_binder_reference_arities_follow_grammar_token_roles() -> None:
    tokenizer = DSLNativeTokenizer.build()
    ids = tokenizer.encode(
        'root = Card([title, body])\ntitle = TextContent(":hero.title")\n'
        'body = Stack([copy], "column")\ncopy = TextContent(":hero.body")',
        add_special=False,
    )
    assert binder_reference_arities(tokenizer, ids) == (
        (tokenizer.bind_id(0), 2),
        (tokenizer.bind_id(1), 0),
        (tokenizer.bind_id(2), 1),
        (tokenizer.bind_id(3), 0),
    )
    prefix = tokenizer.encode("root = Card([title,", add_special=False)
    assert active_declaration_reference_count(tokenizer, prefix) == 1
    assert root_declaration_reference_arity_target(tokenizer, ids) == (2, 3)


def test_lexer_root_reference_arity_trains_and_biases_root_list_paths() -> None:
    model = _model(
        root_reference_arity_loss_weight=1.0,
        root_reference_arity_decode_weight=2.0,
    )
    record = ExampleRecord(
        id="lexer-root-arity",
        prompt="card with title and body",
        openui=(
            "root = Card([title, body])\n"
            'title = TextContent(":hero.title")\n'
            'body = TextContent(":hero.body")'
        ),
        placeholders=[":hero.title", ":hero.body"],
        split="train",
        source="fixture",
    )
    loss = model.training_loss([record])
    loss.backward()

    auxiliary_loss = model.take_detached_auxiliary_loss()
    assert auxiliary_loss is not None
    auxiliary_loss.backward()

    assert model.root_reference_arity_head is not None
    assert model.root_reference_arity_head.weight.grad is not None
    assert model.root_reference_arity_head.weight.grad.abs().sum() > 0
    assert model.last_training_metrics["root_reference_arity_rows"] == 1
    assert model.last_training_metrics["root_reference_arity_accuracy"] == 0.0
    assert model.last_training_metrics["root_reference_arity_classes_mean"] == 3.0
    assert model.root_reference_arity_head.bias.grad is not None
    assert model.root_reference_arity_head.bias.grad[-1] > 0

    tokenizer = model.tokenizer
    with torch.no_grad():
        model.root_reference_arity_head.weight.zero_()
        model.root_reference_arity_head.bias.zero_()
        model.root_reference_arity_head.bias[2] = 4.0
    ctx, ctx_pad = model._encode_context([record.prompt])
    stop = CompletionPath((tokenizer.token_to_id["]"],), "grammar_rsqb")
    continued = CompletionPath((tokenizer.bind_id(2),), "bind_reference")
    one_ref = tokenizer.encode("root = Card([title,", add_special=False)
    two_refs = tokenizer.encode("root = Card([title, body,", add_special=False)

    continue_bias = model._root_reference_arity_path_bias(
        ctx, ctx_pad, one_ref, (stop, continued)
    )
    stop_bias = model._root_reference_arity_path_bias(
        ctx, ctx_pad, two_refs, (stop, continued)
    )
    assert continue_bias is not None and continue_bias[1] > continue_bias[0]
    assert stop_bias is not None and stop_bias[0] > stop_bias[1]


def test_component_edge_supervision_and_parent_conditioned_bias() -> None:
    model = _model(
        component_edge_loss_weight=1.0,
        component_edge_alignment_loss_weight=1.0,
        component_edge_decode_weight=2.0,
    )
    model.train()
    record = ExampleRecord(
        id="component-edge",
        prompt="card with a title",
        openui='root = Card([title])\ntitle = TextContent(":hero.title")',
        placeholders=[":hero.title"],
        split="train",
        source="fixture",
    )
    loss = model.training_loss([record])
    loss.backward()
    assert torch.isfinite(loss)
    assert model.component_edge_head is not None
    assert model.component_edge_head.weight.grad is not None
    assert model.component_edge_head.weight.grad.abs().sum() > 0
    assert model.last_training_metrics["component_edge_loss"] > 0
    assert 0 <= model.last_training_metrics["component_edge_topk_recall"] <= 1
    assert model.last_training_metrics["component_edge_positive_count_mean"] == 1
    assert model.last_training_metrics["component_edge_alignment_rows"] == 1
    assert model.last_training_metrics["component_edge_alignment_loss"] > 0
    assert (
        model.last_training_metrics["component_edge_alignment_unknown_parent_rows"] == 0
    )
    assert 0 <= model.last_training_metrics["component_edge_alignment_accuracy"] <= 1

    tokenizer = model.tokenizer
    components = model._component_inventory_token_ids()
    component_index = {token_id: i for i, token_id in enumerate(components)}
    card = tokenizer.token_to_id["Card"]
    text = tokenizer.token_to_id["TextContent"]
    with torch.no_grad():
        model.component_edge_head.weight.zero_()
        model.component_edge_head.bias.zero_()
        model.component_edge_head.bias[
            component_index[card] * len(components) + component_index[text]
        ] = 3.0
    ctx, ctx_pad = model._encode_context(["card with a title"])
    prefix = tokenizer.encode("root = Card([title])\ntitle =", add_special=False)
    bias = model._component_edge_bias(
        ctx,
        ctx_pad,
        prefix,
        (card, text),
        ("component_bound", "component_bound"),
    )
    assert bias is not None
    assert bias[1] > bias[0]


def test_binder_component_plan_supervises_instances_and_biases_legal_choices() -> None:
    model = _model(
        binder_component_plan_loss_weight=1.0,
        binder_component_plan_decode_weight=2.0,
    )
    model.train()
    record = ExampleRecord(
        id="binder-component-plan",
        prompt="card with title and body",
        openui=(
            "root = Card([title, body])\n"
            'title = TextContent(":hero.title")\n'
            'body = TextContent(":hero.body")'
        ),
        placeholders=[":hero.title", ":hero.body"],
        split="train",
        source="fixture",
    )
    loss = model.training_loss([record])
    loss.backward()
    assert torch.isfinite(loss)
    assert model.binder_component_plan_head is not None
    assert model.binder_component_plan_head.weight.grad is not None
    assert model.binder_component_plan_head.weight.grad.abs().sum() > 0
    assert model.last_training_metrics["binder_component_plan_rows"] == 2
    assert model.last_training_metrics["binder_component_plan_loss"] > 0

    tokenizer = model.tokenizer
    binders = model._binder_component_token_ids()
    components = model._component_inventory_token_ids()
    card = tokenizer.token_to_id["Card"]
    text = tokenizer.token_to_id["TextContent"]
    binder = binders.index(tokenizer.bind_id(1))
    child = components.index(text)
    with torch.no_grad():
        model.binder_component_plan_head.weight.zero_()
        model.binder_component_plan_head.bias.zero_()
        model.binder_component_plan_head.bias[binder * len(components) + child] = 3.0
    ctx, ctx_pad = model._encode_context(["card with title and body"])
    prefix = tokenizer.encode("root = Card([title])\ntitle =", add_special=False)
    bias = model._binder_component_plan_bias(
        ctx,
        ctx_pad,
        prefix,
        (card, text),
        ("component_bound", "component_bound"),
    )
    assert bias is not None
    assert bias[1] > bias[0]


def test_binder_topology_supervises_and_biases_legal_references() -> None:
    model = _model(
        binder_topology_loss_weight=1.0,
        binder_topology_decode_weight=2.0,
    )
    model.train()
    record = ExampleRecord(
        id="binder-topology",
        prompt="card with title and body",
        openui=(
            "root = Stack([group, title, body])\n"
            "group = Card([title, body])\n"
            'title = TextContent(":hero.title")\n'
            'body = TextContent(":hero.body")'
        ),
        placeholders=[":hero.title", ":hero.body"],
        split="train",
        source="fixture",
    )
    loss = model.training_loss([record])
    loss.backward()
    assert torch.isfinite(loss)
    assert model.binder_topology_head is not None
    assert model.binder_topology_head.weight.grad is not None
    assert model.binder_topology_head.weight.grad.abs().sum() > 0
    assert model.last_training_metrics["binder_topology_rows"] > 0
    assert model.last_training_metrics["binder_topology_loss"] > 0

    tokenizer = model.tokenizer
    binders = model._binder_component_token_ids()
    root = binders.index(tokenizer.bind_id(0))
    child = binders.index(tokenizer.bind_id(2))
    with torch.no_grad():
        model.binder_topology_head.weight.zero_()
        model.binder_topology_head.bias.zero_()
        model.binder_topology_head.bias[root * len(binders) + child] = 3.0
    ctx, ctx_pad = model._encode_context(["card with title and body"])
    prefix = tokenizer.encode("root = Card([title,", add_special=False)
    candidates = (tokenizer.bind_id(1), tokenizer.bind_id(2))
    bias = model._binder_topology_bias(
        ctx,
        ctx_pad,
        prefix,
        candidates,
        ("bind_reference_root_children", "bind_reference_root_children"),
    )
    assert bias is not None
    assert bias[1] > bias[0]


def test_binder_arity_supervises_and_biases_continue_stop_paths() -> None:
    model = _model(
        binder_arity_loss_weight=1.0,
        binder_arity_decode_weight=2.0,
    )
    model.train()
    record = ExampleRecord(
        id="binder-arity",
        prompt="card with title and body",
        openui=(
            "root = Card([title, body])\n"
            'title = TextContent(":hero.title")\n'
            'body = TextContent(":hero.body")'
        ),
        placeholders=[":hero.title", ":hero.body"],
        split="train",
        source="fixture",
    )
    loss = model.training_loss([record])
    loss.backward()
    auxiliary_loss = model.take_detached_auxiliary_loss()
    assert auxiliary_loss is not None
    auxiliary_loss.backward()
    assert torch.isfinite(loss)
    assert model.binder_arity_head is not None
    assert model.binder_arity_head.weight.grad is not None
    assert model.binder_arity_head.weight.grad.abs().sum() > 0
    assert model.last_training_metrics["binder_arity_rows"] == 3
    assert model.last_training_metrics["binder_arity_loss"] > 0

    tokenizer = model.tokenizer
    binders = model._binder_component_token_ids()
    root = binders.index(tokenizer.bind_id(0))
    buckets = len(binders) + 1
    with torch.no_grad():
        model.binder_arity_head.weight.zero_()
        model.binder_arity_head.bias.zero_()
        model.binder_arity_head.bias[root * buckets + 2] = 3.0
    ctx, ctx_pad = model._encode_context(["card with title and body"])
    prefix = tokenizer.encode("root = Card([title", add_special=False)
    paths = (
        CompletionPath(
            (tokenizer.token_to_id[","], tokenizer.bind_id(2)),
            "grammar_comma",
        ),
        CompletionPath(
            (tokenizer.token_to_id["]"], tokenizer.token_to_id[")"]),
            "grammar_rsqb",
        ),
    )
    bias = model._binder_arity_path_bias(ctx, ctx_pad, prefix, paths)
    assert bias is not None
    assert bias[0] > bias[1]


def test_tree_verifier_packs_prefix_nodes_and_avoids_full_projection() -> None:
    model = _model()
    model.config.structural_bias = 0.0
    tokenizer = model.tokenizer
    prefix = [tokenizer.bos_id, *tokenizer.encode("root=", add_special=False)]
    paths = (
        CompletionPath(
            (tokenizer.token_to_id["Card"], tokenizer.token_to_id["("]),
            "component",
        ),
        CompletionPath(
            (tokenizer.token_to_id["Stack"], tokenizer.token_to_id["("]),
            "component",
        ),
    )
    ctx, ctx_pad = model._encode_context(["card"])
    canvas = model._compiler_canvas(prefix, 24)
    full = model.denoiser(
        canvas,
        ctx,
        pad_id=tokenizer.pad_id,
        ctx_pad_mask=ctx_pad,
    )[0, len(prefix)]
    expected = max(paths, key=lambda path: float(full[path.token_ids[0]].item()))
    with collect_decode_stats() as stats:
        selected = model._select_compiler_path(
            prefix, paths, ctx, ctx_pad, 24, tree=True
        )
    assert selected in {path.token_ids for path in paths}
    assert selected == expected.token_ids
    assert stats.trie_nodes >= 3
    assert stats.restricted_projections >= 1
    assert stats.full_projections == 0
    assert stats.constrained_selection_traces[0]["phase"] == "compiler_tree"
    assert stats.constrained_selection_traces[0]["top_candidates"]
    assert (
        "first_edge_score" in stats.constrained_selection_traces[0]["top_candidates"][0]
    )


def test_tree_verifier_records_exact_branch_support_for_event_mining(
    monkeypatch,
) -> None:
    model = _model()
    model.config.structural_bias = 0.0
    tokenizer = model.tokenizer
    prefix = [tokenizer.bos_id, *tokenizer.encode("root=", add_special=False)]
    paths = (
        CompletionPath(
            (tokenizer.token_to_id["Card"], tokenizer.token_to_id["("]),
            "component",
        ),
        CompletionPath(
            (tokenizer.token_to_id["Stack"], tokenizer.token_to_id["("]),
            "component",
        ),
    )
    ctx, ctx_pad = model._encode_context(["card"])
    original_project = model.denoiser.project

    def project(hidden, candidate_ids=None):
        scores = original_project(hidden, candidate_ids)
        if candidate_ids is None:
            scores = scores.clone()
            scores[tokenizer.mask_id] = scores.max() + 100.0
        return scores

    monkeypatch.setattr(model.denoiser, "project", project)
    recorder = DecodeTraceRecorder(record_support=True)
    model.trace_recorder = recorder
    selected = model._select_compiler_path(prefix, paths, ctx, ctx_pad, 24, tree=True)
    model.trace_recorder = None

    trace = recorder.finalize(
        record_id="record-1",
        context_text="card",
        policy_checkpoint_sha="checkpoint-sha",
        tokenizer_sha="tokenizer-sha",
        decode_config_hash="decode-sha",
        seed=0,
    )
    trace["trajectory_id"] = "trajectory-1"
    commits = [commit for step in trace["steps"] for commit in step["commits"]]
    assert commits
    assert commits[0]["id"] == selected[0]
    assert commits[0]["raw_id"] == tokenizer.mask_id
    assert set(commits[0]["allowed_id_set"]) == {
        tokenizer.token_to_id["Card"],
        tokenizer.token_to_id["Stack"],
    }
    assert commits[0]["pre_canvas"][: len(prefix)] == prefix
    events = events_from_trace(trace)
    assert len(events) == 1
    assert events[0].good_token_ids == (selected[0],)
    assert events[0].bad_token_ids == (tokenizer.mask_id,)


def test_maskgit_fallback_keeps_compiler_prefix_visible() -> None:
    model = _model()
    tokenizer = model.tokenizer
    seed = [tokenizer.bos_id, *tokenizer.encode("root=Stack(", add_special=False)]
    ctx, ctx_pad = model._encode_context(["card"])
    text = model._generate_maskgit_one(
        ctx,
        ctx_pad,
        24,
        use_grammar=False,
        seed_ids=seed,
    )
    # The lexer stores root as binding zero internally, but public output is
    # canonical OpenUI and must preserve the seeded root component surface.
    assert text.startswith("root = Stack(")


def test_compiler_decode_is_opt_in() -> None:
    assert TwoTowerConfig().compiler_decode_mode == "off"
    assert TwoTowerConfig().compiler_search_mode == "greedy"


def test_compiler_decode_reserves_room_beyond_predicted_length(monkeypatch) -> None:
    model = _model()
    model.config.compiler_decode_mode = "tree"
    model.config.grammar_ltr_primary = True
    model.config.grammar_draft_window = 5
    monkeypatch.setattr(model, "_predict_target_lengths", lambda *_: [12])
    observed: list[int] = []

    def decode(_ctx, _ctx_pad, length: int) -> torch.Tensor:
        observed.append(length)
        return torch.full((1, length), model.tokenizer.eos_id, dtype=torch.long)

    monkeypatch.setattr(model, "_greedy_ltr_decode_batch", decode)
    monkeypatch.setattr(model, "_decode_ids", lambda _ids: "root = Stack([])")
    monkeypatch.setattr(model, "_ensure_valid_openui", lambda text, *_a, **_k: text)
    assert model.generate_batch_requests([GenerationRequest(prompt="card")]) == [
        "root = Stack([])"
    ]
    assert observed == [17]


def test_lattice_search_rejects_unknown_mode() -> None:
    model = _model()
    model.config.compiler_search_mode = "unknown"
    ctx, ctx_pad = model._encode_context(["card"])
    with pytest.raises(ValueError, match="must be greedy, lattice, ptrm, or gram"):
        model._compiler_ltr_decode_one(
            ctx, ctx_pad, 24, mode="tree", slot_contract=None
        )


def test_lattice_search_matches_greedy_without_conflict() -> None:
    greedy = _model()
    ctx, ctx_pad = greedy._encode_context(["card"])
    expected = greedy._compiler_ltr_decode_one(
        ctx, ctx_pad, 24, mode="tree", slot_contract=None
    )
    model = _model()
    model.config.compiler_search_mode = "lattice"
    ctx, ctx_pad = model._encode_context(["card"])
    with collect_decode_stats() as stats:
        ids = model._compiler_ltr_decode_one(
            ctx, ctx_pad, 24, mode="tree", slot_contract=None
        )
    assert torch.equal(ids, expected)
    assert stats.compiler_lattice_states > 0
    assert stats.compiler_lattice_candidates >= stats.compiler_lattice_states


def test_ptrm_trajectory_policy_is_seed_reproducible() -> None:
    model = _model()
    model.config.compiler_search_mode = "ptrm"
    model.config.compiler_search_trigger = "always"
    model.config.compiler_search_width = 4
    model.config.compiler_search_noise = 1.0
    ctx, ctx_pad = model._encode_context(["card"])
    with collect_decode_stats() as left_stats:
        left = model._compiler_ltr_decode_one(
            ctx, ctx_pad, 24, mode="tree", slot_contract=None
        )
    with collect_decode_stats() as right_stats:
        right = model._compiler_ltr_decode_one(
            ctx, ctx_pad, 24, mode="tree", slot_contract=None
        )
    assert torch.equal(left, right)
    assert left_stats.compiler_lattice_trajectory_triggers > 0
    assert left_stats.compiler_lattice_trajectories == (
        right_stats.compiler_lattice_trajectories
    )


# --- VSS0-02: reason-coded constraint evidence -------------------------------


def _evidence_parity_prefixes(tokenizer) -> list[list[int]]:
    return [
        [],
        [tokenizer.bos_id],
        [tokenizer.bos_id, tokenizer.token_to_id["NL"]],
        [tokenizer.bos_id, *tokenizer.encode("root=Stack([", add_special=False)],
        [tokenizer.bos_id, *tokenizer.encode("root=Stack([b1,", add_special=False)],
    ]


def test_constraint_evidence_off_by_default_and_observational_when_on() -> None:
    tokenizer = DSLNativeTokenizer.build()
    for prefix in _evidence_parity_prefixes(tokenizer):
        off = build_completion_forest(tokenizer, list(prefix))
        on = build_completion_forest(tokenizer, list(prefix), explain=True)
        # Default path stays silent and byte-for-byte unchanged.
        assert off.evidence == ()
        assert off.evidence_summary is None
        assert on.candidate_ids == off.candidate_ids
        assert on.coverage == off.coverage
        assert on.terminals == off.terminals
        assert on.paths == off.paths
        # Explanation is populated, self-consistent, and honest about coverage.
        assert on.evidence
        assert on.evidence_summary is not None
        assert on.evidence_summary.coverage == on.coverage
        assert all(record.reason_code for record in on.evidence)
        cover = [e for e in on.evidence if e.stage is ConstraintStage.COVERAGE]
        assert len(cover) == 1
        assert cover[0].admitted is (on.coverage == "complete")
        assert cover[0].reason_code == f"coverage_{on.coverage}"
        # Admitted evidence reproduces exactly the emitted candidate set.
        admitted = {
            e.candidate_id
            for e in on.evidence
            if e.admitted and e.candidate_id is not None
        }
        assert admitted == set(on.candidate_ids)


def test_constraint_evidence_localizes_binder_and_min_content() -> None:
    tokenizer = DSLNativeTokenizer.build()
    # A forward binder that is out of scope is excluded at the BINDING stage.
    prefix = [tokenizer.bos_id, *tokenizer.encode("root=Stack([", add_special=False)]
    forest = build_completion_forest(tokenizer, prefix, explain=True)
    assert tokenizer.bind_id(2) not in forest.candidate_ids
    binder_excluded = [
        e
        for e in forest.evidence
        if e.candidate_id == tokenizer.bind_id(2) and not e.admitted
    ]
    assert binder_excluded
    assert any(e.stage is ConstraintStage.BINDING for e in binder_excluded)

    # Minimum-content EOS withholding is distinguishable from grammar rejection.
    contract = [":hero.title"]
    complete = [
        tokenizer.bos_id,
        *tokenizer.encode('root=TextContent(":hero.title")', add_special=False),
    ]
    unmet = build_completion_forest(
        tokenizer, complete, slot_contract=contract, min_content=2, explain=True
    )
    assert tokenizer.eos_id not in unmet.candidate_ids
    eos_evidence = [e for e in unmet.evidence if e.candidate_id == tokenizer.eos_id]
    assert eos_evidence
    assert any(
        e.stage is ConstraintStage.MIN_CONTENT
        and e.reason_code == "eos_withheld_min_content"
        for e in eos_evidence
    )
    # It is not misattributed to a grammar rejection.
    assert not any(e.stage is ConstraintStage.GRAMMAR for e in eos_evidence)

    met = build_completion_forest(
        tokenizer, complete, slot_contract=contract, min_content=1, explain=True
    )
    assert tokenizer.eos_id in met.candidate_ids
    assert any(
        e.candidate_id == tokenizer.eos_id
        and e.admitted
        and e.stage is ConstraintStage.TERMINAL
        for e in met.evidence
    )


def test_constraint_evidence_schema_stage_and_partial_coverage(monkeypatch) -> None:
    from slm_training.dsl.grammar.fastpath import compiler_draft

    tokenizer = DSLNativeTokenizer.build()
    schema = {
        "properties": {"Stack": {}},
        "$defs": {
            "Stack": {
                "properties": {
                    "children": {"type": "array"},
                    "direction": {"enum": ["row", "column"]},
                }
            }
        },
    }
    monkeypatch.setattr(compiler_draft, "_official_schema", lambda: schema)
    prefix = tokenizer.encode("root=Stack([],", add_special=False)
    forest = build_completion_forest(tokenizer, prefix, explain=True)
    assert set(forest.candidate_ids) == {
        tokenizer.token_to_id["STR:row"],
        tokenizer.token_to_id["STR:column"],
    }
    assert any(
        e.stage is ConstraintStage.SCHEMA and not e.admitted for e in forest.evidence
    )
    assert forest.evidence_summary.coverage == forest.coverage

    # Schema needed but unavailable → coverage is not certified complete, and the
    # COVERAGE record refuses to serialize a partial forest as an exact proof.
    monkeypatch.setattr(compiler_draft, "_official_schema", lambda: None)
    partial = build_completion_forest(
        tokenizer,
        tokenizer.encode("root=TextContent(", add_special=False),
        explain=True,
    )
    assert partial.coverage != "complete"
    cover = [e for e in partial.evidence if e.stage is ConstraintStage.COVERAGE]
    assert len(cover) == 1
    assert cover[0].admitted is False
    assert cover[0].reason_code == f"coverage_{partial.coverage}"


def test_constraint_evidence_is_deterministic_and_json_roundtrips() -> None:
    tokenizer = DSLNativeTokenizer.build()
    prefix = [tokenizer.bos_id, *tokenizer.encode("root=Stack([b1,", add_special=False)]
    first = build_completion_forest(tokenizer, list(prefix), explain=True)
    second = build_completion_forest(tokenizer, list(prefix), explain=True)
    assert first.evidence == second.evidence
    assert first.evidence_summary == second.evidence_summary
    for record in first.evidence:
        restored = ConstraintEvidence.from_dict(
            json.loads(json.dumps(record.as_dict()))
        )
        assert restored == record
    # The whole-forest view is JSON-serializable.
    json.dumps(first.evidence_as_json())


def test_required_slot_margin_bias_floors_only_still_missing_slots() -> None:
    """E626: floor the best-scoring legal slot candidate that is genuinely

    still missing from the prefix; already-filled slots and the fully-covered
    case are both no-ops, and the default-off weight never fires.
    """
    model = _model(output_tokenizer="choice", required_slot_margin_decode_weight=2.0)
    tokenizer = model.tokenizer
    slot0 = tokenizer.sym_id(0)
    slot1 = tokenizer.sym_id(1)
    button_id = tokenizer.token_to_id["+Button"]
    slot_contract = [":status.title", ":status.body"]
    candidates = (slot0, slot1, button_id)
    scores = torch.tensor([9.0, 2.0, 10.0])

    # slot0 already appears in the prefix; only slot1 is still missing.
    prefix = [tokenizer.bos_id, slot0]
    bias = model._required_slot_margin_bias(prefix, candidates, scores, slot_contract)

    assert bias is not None
    assert bias.tolist() == [0.0, 10.0, 0.0]

    # Once every visible slot has appeared, the lever no-ops (nothing missing).
    assert (
        model._required_slot_margin_bias(
            [*prefix, slot1], candidates, scores, slot_contract
        )
        is None
    )

    # Default-off weight never fires, even with missing slots present.
    off_model = _model(
        output_tokenizer="choice", required_slot_margin_decode_weight=0.0
    )
    assert (
        off_model._required_slot_margin_bias(prefix, candidates, scores, slot_contract)
        is None
    )

    # No slot contract -> no-op regardless of weight.
    assert model._required_slot_margin_bias(prefix, candidates, scores, None) is None


def test_native_slot_contract_excludes_already_emitted_marker() -> None:
    from slm_training.models.grammar import contract_allowed_token_ids

    model = _model()
    tokenizer = model.tokenizer
    slot0 = tokenizer.sym_id(0)
    slot1 = tokenizer.sym_id(1)

    assert contract_allowed_token_ids(
        tokenizer,
        [tokenizer.bos_id, slot0],
        [":status.title", ":status.body"],
    ) == {slot1}


def test_required_slot_margin_bias_excludes_frame_depth_zero() -> None:
    """E628: exclude the root/top-level statement position from the floor.

    E627 traced margin=6's Dashboard regression to every hijack firing at
    ``frame_depth == 0`` -- a fresh top-level statement's value, where the
    grammar legally allows a bare visible-slot token as an alternative to
    opening a real component. Passing a ``state`` whose ``frames`` is empty
    (frame_depth 0) must now make the bias a no-op entirely, even though the
    same missing-slot/candidate/score setup would otherwise floor it (as
    ``test_required_slot_margin_bias_floors_only_still_missing_slots`` proves
    above). Once inside a real component/object frame (``frames`` non-empty,
    frame_depth >= 1) the bias still fires exactly as before -- this lever
    should keep flooring slots-as-arguments, only no longer compete for the
    root/top-level choice.
    """
    from types import SimpleNamespace

    model = _model(output_tokenizer="choice", required_slot_margin_decode_weight=2.0)
    tokenizer = model.tokenizer
    slot0 = tokenizer.sym_id(0)
    slot1 = tokenizer.sym_id(1)
    button_id = tokenizer.token_to_id["+Button"]
    slot_contract = [":status.title", ":status.body"]
    candidates = (slot0, slot1, button_id)
    scores = torch.tensor([9.0, 2.0, 10.0])
    prefix = [tokenizer.bos_id, slot0]

    # frame_depth == 0 (no frame open yet): the bias must not fire at all,
    # regardless of how genuinely missing slot1 is.
    root_state = SimpleNamespace(frames=[])
    assert (
        model._required_slot_margin_bias(
            prefix, candidates, scores, slot_contract, state=root_state
        )
        is None
    )

    # frame_depth == 1 (inside an open component frame) at a content-flagged
    # argument position (E630's schema-position gate, see
    # ``test_required_slot_margin_bias_excludes_non_content_schema_positions``
    # below): unchanged behavior, matching the no-state-passed case exactly.
    inner_frame = SimpleNamespace(
        kind="component",
        expr_type="element:Card",
        phase="args",
        arg_index=0,
        schemas=({"x-openui-placeholder": True},),
    )
    inner_state = SimpleNamespace(frames=[inner_frame])
    bias = model._required_slot_margin_bias(
        prefix, candidates, scores, slot_contract, state=inner_state
    )
    assert bias is not None
    assert bias.tolist() == [0.0, 10.0, 0.0]

    # state=None (unknown depth, e.g. a lower-level direct call) preserves
    # the pre-E628 behavior of firing unconditionally.
    assert (
        model._required_slot_margin_bias(
            prefix, candidates, scores, slot_contract, state=None
        )
        is not None
    )


def test_required_slot_margin_bias_excludes_non_content_schema_positions() -> None:
    """E630: root-causing rico_eval_test_25's frame_depth>=1 over-stuffing.

    E629's widened-suite sweep found a second failure mode distinct from
    E627/E628's frame_depth==0 root hijack: at margin>=1, one ``Button``
    absorbed 5 still-missing required slots across all 5 of its positional
    arguments (``label``, ``action``, ``variant``, ``type``, ``size``), not
    just the content-flagged ``label``. A live trace (E630) confirmed the
    mechanism directly: ``_required_slot_margin_bias``'s floor
    (``old_max + margin``) is computed relative to the *current* best legal
    score -- i.e. after ``_schema_value_bias``/``_schema_opaque_bias`` have
    already discouraged a placeholder there and ``_schema_enum_close_bias``/
    ``_schema_opaque_close_bias`` have already preferred closing instead --
    so the floor wins against all four at *any* margin > 0, not only a large
    one, because all four run earlier in the same per-position bias stack.
    Gating the fire to exactly the positions ``_schema_role_slot_bias``
    already treats as slot-eligible removes this: a ``component`` frame's
    active argument must be ``x-openui-placeholder``-flagged (a content
    property), and an ``object`` frame's active property schema must be able
    to reach a visible slot at all.
    """
    from types import SimpleNamespace

    model = _model(output_tokenizer="choice", required_slot_margin_decode_weight=2.0)
    tokenizer = model.tokenizer
    slot0 = tokenizer.sym_id(0)
    slot1 = tokenizer.sym_id(1)
    button_id = tokenizer.token_to_id["+Button"]
    slot_contract = [":status.title", ":status.body"]
    candidates = (slot0, slot1, button_id)
    scores = torch.tensor([9.0, 2.0, 10.0])
    prefix = [tokenizer.bos_id, slot0]

    # component frame, non-content (enum) argument position: no fire, even
    # though slot1 is genuinely still missing and legal there -- matches
    # Button.variant/type/size in the traced rico_eval_test_25 regression.
    enum_frame = SimpleNamespace(
        kind="component",
        expr_type="element:Button",
        phase="args",
        arg_index=1,
        schemas=(
            {"x-openui-placeholder": True},
            {"type": "string", "enum": ["primary", "secondary"]},
        ),
    )
    assert (
        model._required_slot_margin_bias(
            prefix,
            candidates,
            scores,
            slot_contract,
            state=SimpleNamespace(frames=[enum_frame]),
        )
        is None
    )

    # component frame, the content (label) argument position: fires exactly
    # as before -- this lever still floors slots as content-property fills.
    content_frame = SimpleNamespace(
        kind="component",
        expr_type="element:Button",
        phase="args",
        arg_index=0,
        schemas=(
            {"x-openui-placeholder": True},
            {"type": "string", "enum": ["primary", "secondary"]},
        ),
    )
    bias = model._required_slot_margin_bias(
        prefix,
        candidates,
        scores,
        slot_contract,
        state=SimpleNamespace(frames=[content_frame]),
    )
    assert bias is not None
    assert bias.tolist() == [0.0, 10.0, 0.0]

    # object frame, a property whose schema cannot reach a visible slot
    # (e.g. a plain boolean): no fire.
    boolean_object_frame = SimpleNamespace(
        kind="object", arg_index=0, schemas=({"type": "boolean"},)
    )
    assert (
        model._required_slot_margin_bias(
            prefix,
            candidates,
            scores,
            slot_contract,
            state=SimpleNamespace(frames=[boolean_object_frame]),
        )
        is None
    )

    # object frame, a property whose schema can reach a visible slot: fires.
    reachable_object_frame = SimpleNamespace(
        kind="object", arg_index=0, schemas=({"x-openui-placeholder": True},)
    )
    bias = model._required_slot_margin_bias(
        prefix,
        candidates,
        scores,
        slot_contract,
        state=SimpleNamespace(frames=[reachable_object_frame]),
    )
    assert bias is not None
    assert bias.tolist() == [0.0, 10.0, 0.0]

    # A frame kind this bias never previously scoped (e.g. "variadic" array
    # items) stays permissive -- this fix only targets the newly-traced
    # component/object argument-position failure mode.
    variadic_frame = SimpleNamespace(kind="variadic", expr_type="array", arg_index=0)
    bias = model._required_slot_margin_bias(
        prefix,
        candidates,
        scores,
        slot_contract,
        state=SimpleNamespace(frames=[variadic_frame]),
    )
    assert bias is not None
    assert bias.tolist() == [0.0, 10.0, 0.0]


def test_required_slot_margin_trace_flags_a_root_level_component_hijack() -> None:
    """E627 root-cause instrumentation for E626's open margin=6 regression.

    When a still-missing slot candidate is legal at the *same* decode
    position as a real component-opening candidate (frame_depth 0, i.e. no
    component frame has been opened yet), a large-enough margin can floor
    the slot token above the component token and win the position's argmax
    -- this is the mechanism E626 observed collapsing Dashboard to a bare
    ``Button``. The trace must surface this precisely: ``frame_depth == 0``,
    ``chosen_kind == "sym"``, and ``hijacked_non_slot_candidate`` true
    because the pre-bias argmax was a non-slot (component) candidate.

    This test exercises ``_record_required_slot_margin_trace`` directly with a
    hand-built ``margin_bias`` -- it stays a valid unit test of the trace
    function's own labeling logic even after E628 makes this exact scenario
    unreachable in production (``_required_slot_margin_bias`` now returns
    ``None`` at ``frame_depth == 0``, so this call site is never reached with
    a live margin_bias there anymore; see
    ``test_required_slot_margin_bias_excludes_frame_depth_zero``).
    """
    from types import SimpleNamespace

    from slm_training.models.decode_stats import DecodeStats

    model = _model(
        output_tokenizer="choice",
        required_slot_margin_decode_weight=6.0,
    )
    tokenizer = model.tokenizer
    slot_id = tokenizer.sym_id(3)
    button_id = tokenizer.token_to_id["+Button"]
    candidate_ids = (button_id, slot_id)
    candidate_kinds = ("component_root_or_bound", "sym")
    stats = DecodeStats()

    trace = model._record_required_slot_margin_trace(
        stats,
        row=0,
        position=1,
        state=SimpleNamespace(frames=[]),
        candidate_ids=candidate_ids,
        candidate_kinds=candidate_kinds,
        scores_before=torch.tensor([11.8, 10.3]),
        margin_bias=torch.tensor([0.0, 7.5]),
        scores_after=torch.tensor([11.8, 17.8]),
    )

    assert trace is not None
    assert trace["phase"] == "required_slot_margin"
    assert trace["frame_depth"] == 0
    assert trace["before_token"] == "+Button"
    assert trace["before_kind"] == "component_root_or_bound"
    assert trace["chosen_token"] == tokenizer.id_to_token[slot_id]
    assert trace["chosen_kind"] == "sym"
    assert trace["choice_changed"] is True
    assert trace["hijacked_non_slot_candidate"] is True
    assert stats.constrained_selection_traces == [trace]

    # Inside an open component frame, a slot-vs-slot swap is not a hijack:
    # the pre-bias argmax was already a slot candidate.
    open_frame = SimpleNamespace(
        kind="component", expr_type="element:Button", phase="args", arg_index=0
    )
    inner_trace = model._record_required_slot_margin_trace(
        stats,
        row=0,
        position=7,
        state=SimpleNamespace(frames=[open_frame]),
        candidate_ids=(slot_id, tokenizer.sym_id(4)),
        candidate_kinds=("sym", "sym"),
        scores_before=torch.tensor([9.0, 8.0]),
        margin_bias=torch.tensor([0.0, 2.0]),
        scores_after=torch.tensor([9.0, 10.0]),
    )
    assert inner_trace is not None
    assert inner_trace["frame_depth"] == 1
    assert inner_trace["choice_changed"] is True
    assert inner_trace["hijacked_non_slot_candidate"] is False


def test_schema_role_slot_bias_skips_masked_bound_role() -> None:
    from slm_training.dsl.production_codec import OPEN_PREFIX
    from slm_training.models.choice_tokenizer import ChoiceDecodeState

    model = _model(output_tokenizer="choice", schema_role_slot_decode_weight=8.0)
    tokenizer = model.tokenizer
    state = ChoiceDecodeState(tokenizer, slot_count=2)
    assert state.advance_id(tokenizer.token_to_id[f"{OPEN_PREFIX}Button"])
    alt_id = tokenizer.sym_id(0)
    cta_id = tokenizer.sym_id(1)

    bias = model._schema_role_slot_bias(
        state,
        (alt_id, cta_id),
        torch.tensor([20.0, float("-inf")]),
        [":gallery.alt", ":gallery.cta"],
        {
            ":gallery.alt": ("ImageGallery",),
            ":gallery.cta": ("Button",),
        },
        prefix=[],
        role_bindings={"Button": (":gallery.cta",)},
    )

    assert bias is None
