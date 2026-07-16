"""Unit tests for grammar fast-path (force-emit + MaskGIT admit)."""

from __future__ import annotations

from slm_training.dsl.schema import ExampleRecord
from slm_training.dsl.grammar.fastpath import (
    OpenUIIncrementalEngine,
    admit_fill,
    draft_forced_ids,
    engine_for_dsl,
    force_next_token_id,
)
from slm_training.models.grammar import force_emit_token_id, pick_constrained_token
from slm_training.models.tokenizer import OpenUITokenizer
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel

SAMPLE = 'root = Card(":t.x")\n'


def _tok() -> OpenUITokenizer:
    return OpenUITokenizer.build([SAMPLE, 'root = Row(":j")\n', "root = Stack([hero])\n"])


def test_engine_force_equal_after_name() -> None:
    eng = OpenUIIncrementalEngine()
    assert eng.set_prefix("root")
    assert eng.is_deterministic_next() == "="


def test_engine_force_lpar_after_component() -> None:
    eng = OpenUIIncrementalEngine()
    assert eng.set_prefix("root = Row")
    assert eng.is_deterministic_next() == "("


def test_force_next_token_id_maps_equal() -> None:
    from slm_training.models.dsl_tokenizer import DSLNativeTokenizer

    tok = DSLNativeTokenizer.build()
    eng = engine_for_dsl("openui")
    assert eng is not None
    tid = force_next_token_id(eng, tok, "root")
    assert tid is not None
    assert tok.id_to_token[tid] == "="


def test_force_emit_token_id_via_grammar_helper() -> None:
    tok = _tok()
    ids = tok.encode("root", add_special=False)
    forced = force_emit_token_id(tok, ids)
    assert forced is not None
    assert tok.id_to_token[forced] == "="


def test_pick_constrained_honors_forced_id() -> None:
    import torch

    tok = _tok()
    equal_id = tok.token_to_id["="]
    logits = torch.zeros(tok.vocab_size)
    # Prefer an illegal punctuation token as argmax so force must win.
    # (NAME gluing like "root"+"Card"→"rootCard" is DFA-legal as a longer NAME.)
    wrong = tok.token_to_id.get("]", equal_id)
    logits[wrong] = 10.0
    prefix = tok.encode("root", add_special=False)
    choice = pick_constrained_token(
        logits, tok, prefix, forced_token_id=equal_id
    )
    assert choice == equal_id


def test_pick_constrained_requires_root_at_first_significant_token() -> None:
    import torch

    tok = _tok()
    logits = torch.full((tok.vocab_size,), -20.0)
    logits[tok.token_to_id["hero"]] = 50.0
    logits[tok.token_to_id["root"]] = 1.0
    choice = pick_constrained_token(logits, tok, [], top_k=8)
    assert choice == tok.token_to_id["root"]


def test_lexer_root_round_trips_and_is_first_token_legal() -> None:
    import torch

    from slm_training.models.dsl_tokenizer import DSLNativeTokenizer

    tok = DSLNativeTokenizer.build()
    ids = tok.encode("root = Stack([])", add_special=False)
    assert tok.decode(ids).startswith("root =")
    logits = torch.full((tok.vocab_size,), -20.0)
    logits[tok.bind_id(0)] = 50.0
    logits[tok.token_to_id["Stack"]] = 1.0
    choice = pick_constrained_token(logits, tok, [], top_k=8)
    assert choice == tok.bind_id(0)


def test_grammar_state_uses_surface_text_for_lexer_ids() -> None:
    from slm_training.models.grammar import make_grammar_state
    from slm_training.models.dsl_tokenizer import DSLNativeTokenizer

    tok = DSLNativeTokenizer.build()
    state = make_grammar_state()
    state.advance_token(tok, tok.bind_id(0))
    assert state.prefix_text == "root"
    assert state.engine is not None
    assert state.engine._prefix == "root"


def test_grammar_state_ignores_bos_before_root_pick() -> None:
    import torch

    from slm_training.models.dsl_tokenizer import DSLNativeTokenizer
    from slm_training.models.grammar import make_grammar_state

    tok = DSLNativeTokenizer.build()
    state = make_grammar_state()
    state.advance_token(tok, tok.bos_id)
    assert state.prefix_text == ""

    logits = torch.full((tok.vocab_size,), -20.0)
    logits[tok.state_id(44)] = 50.0
    assert pick_constrained_token(
        logits, tok, [tok.bos_id], top_k=8, state=state
    ) == tok.bind_id(0)


def test_native_slot_contract_preserves_assignment_and_component() -> None:
    import torch

    from slm_training.models.dsl_tokenizer import DSLNativeTokenizer
    from slm_training.models.grammar import force_emit_token_id, make_grammar_state

    tok = DSLNativeTokenizer.build()
    prefix = tok.encode("root", add_special=True)[:-1]
    state = make_grammar_state()
    forced = force_emit_token_id(tok, prefix, state=state)
    logits = torch.full((tok.vocab_size,), -20.0)
    logits[tok.sym_id(0)] = 50.0
    assert pick_constrained_token(
        logits,
        tok,
        prefix,
        top_k=8,
        forced_token_id=forced,
        slot_contract=[":only.slot"],
        state=state,
    ) == tok.token_to_id["="]

    prefix = [tok.bos_id, tok.bind_id(0), tok.token_to_id["="]]
    state = make_grammar_state()
    for tid in prefix:
        state.advance_token(tok, tid)
    logits[tok.token_to_id["B:6d"]] = 60.0
    logits[tok.token_to_id["TextArea"]] = 1.0
    assert pick_constrained_token(
        logits,
        tok,
        prefix,
        top_k=8,
        slot_contract=[":only.slot"],
        state=state,
    ) == tok.token_to_id["TextArea"]


def test_lexer_literal_bytes_are_grammar_admitted() -> None:
    import torch

    from slm_training.models.dsl_tokenizer import DSLNativeTokenizer

    tok = DSLNativeTokenizer.build()
    prefix = tok.encode('root = Separator("', add_special=False)
    byte = tok.token_to_id["B:68"]
    logits = torch.full((tok.vocab_size,), -20.0)
    logits[byte] = 50.0
    choice = pick_constrained_token(logits, tok, prefix, top_k=8)
    assert choice == byte


def test_structural_preference_does_not_override_confident_binder() -> None:
    import torch

    from slm_training.models.dsl_tokenizer import DSLNativeTokenizer

    tok = DSLNativeTokenizer.build()
    prefix = tok.encode("root = Stack([", add_special=False)
    binder = tok.bind_id(1)
    logits = torch.full((tok.vocab_size,), -20.0)
    logits[binder] = 50.0
    choice = pick_constrained_token(logits, tok, prefix, top_k=8)
    assert choice == binder


def test_lexer_newline_is_probed_as_surface_newline() -> None:
    import torch

    from slm_training.models.dsl_tokenizer import DSLNativeTokenizer

    tok = DSLNativeTokenizer.build()
    prefix = tok.encode("root = Stack([])", add_special=False)
    logits = torch.full((tok.vocab_size,), -20.0)
    logits[tok.token_to_id["NL"]] = 50.0
    choice = pick_constrained_token(logits, tok, prefix, top_k=8)
    assert choice == tok.token_to_id["NL"]


def test_singleton_admission_bypasses_probe(monkeypatch) -> None:
    import torch

    import slm_training.models.grammar as grammar
    from slm_training.models.decode_stats import collect_decode_stats

    tok = _tok()
    prefix = tok.encode("root", add_special=False)
    logits = torch.full((tok.vocab_size,), -20.0)
    logits[tok.token_to_id["="]] = 50.0
    monkeypatch.setattr(
        grammar,
        "dfa_admits_token",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("singleton admission must not be probed")
        ),
    )
    with collect_decode_stats() as stats:
        assert pick_constrained_token(logits, tok, prefix, top_k=8) == tok.token_to_id["="]
    assert stats.constrained_last_legal_candidates == 1


def test_empty_native_prefix_selects_only_legal_root_binding() -> None:
    import torch

    from slm_training.models.dsl_tokenizer import DSLNativeTokenizer

    tok = DSLNativeTokenizer.build()
    logits = torch.full((tok.vocab_size,), -20.0)
    logits[tok.bind_id(0)] = 50.0
    assert pick_constrained_token(logits, tok, [], top_k=8) == tok.bind_id(0)


def test_semantic_guards_run_before_singleton_bypass() -> None:
    import torch

    from slm_training.models.dsl_tokenizer import DSLNativeTokenizer

    tok = DSLNativeTokenizer.build()
    logits = torch.full((tok.vocab_size,), -20.0)
    logits[tok.token_to_id["NL"]] = 50.0
    logits[tok.token_to_id["="]] = 1.0
    prefix = tok.encode("root", add_special=False)
    assert pick_constrained_token(logits, tok, prefix, top_k=8) != tok.token_to_id["NL"]

    logits = torch.full((tok.vocab_size,), -20.0)
    logits[tok.token_to_id[")"]] = 50.0
    prefix = tok.encode("root = Form(", add_special=False)
    assert pick_constrained_token(logits, tok, prefix, top_k=8) != tok.token_to_id[")"]


def test_admit_fill_accepts_partial_with_holes() -> None:
    tok = _tok()
    eng = OpenUIIncrementalEngine()
    ids = tok.encode(SAMPLE, add_special=True)
    # Mask the quoted placeholder span (tokenizer v2 splits ":t.x" into pieces).
    # Left span `root = Card(` must remain a valid incomplete prefix.
    quote_id = tok.token_to_id['"']
    first = ids.index(quote_id)
    second = ids.index(quote_id, first + 1)
    for pos in range(first, second + 1):
        ids[pos] = tok.mask_id
    assert admit_fill(eng, tok, ids) is True


def test_allowed_id_set_expands_components() -> None:
    from slm_training.dsl.grammar.fastpath.token_map import allowed_id_set

    tok = _tok()
    eng = OpenUIIncrementalEngine()
    assert eng.set_prefix("root=")
    allowed = allowed_id_set(tok, eng.next_terminals())
    assert allowed is not None
    assert tok.token_to_id["Stack"] in allowed
    assert tok.token_to_id["Card"] in allowed
    assert tok.token_to_id["="] not in allowed


def test_cached_native_masks_intersect_active_symbols() -> None:
    from slm_training.dsl.grammar.fastpath.token_map import allowed_id_set
    from slm_training.models.dsl_tokenizer import DSLNativeTokenizer

    tok = DSLNativeTokenizer.build()
    terminals = frozenset({"STRING"})
    active = {tok.sym_id(1)}
    uncached = allowed_id_set(tok, terminals)
    cached = allowed_id_set(
        tok,
        terminals,
        active_dynamic_ids=active,
        use_cache=True,
    )
    assert uncached is not None and cached is not None
    assert tok.sym_id(1) in cached
    assert tok.sym_id(0) not in cached
    assert cached - tok.kind_ids("sym") == uncached - tok.kind_ids("sym")


def test_completion_bound_is_conservative() -> None:
    engine = OpenUIIncrementalEngine()
    assert engine.minimum_completion_tokens("root = Stack([])") == 0
    assert engine.minimum_completion_tokens("root = ") is None


def test_dsl_native_terminal_map_covers_v05_surface() -> None:
    from slm_training.dsl.grammar.fastpath.token_map import allowed_id_set
    from slm_training.models.dsl_tokenizer import DSLNativeTokenizer

    tok = DSLNativeTokenizer.build()
    terminals = frozenset(
        {
            "LBRACE",
            "RBRACE",
            "COLON",
            "QMARK",
            "DOT",
            "PLUS",
            "__ANON_4",
            "BUILTIN",
            "STATE_NAME",
            "NULL",
        }
    )
    allowed = allowed_id_set(tok, terminals)
    assert allowed is not None
    for token in ("{", "}", ":", "?", ".", "+", ">=", "@Run", "null"):
        assert tok.token_to_id[token] in allowed
    assert tok.state_id(0) in allowed


def test_pick_constrained_rejects_double_equal() -> None:
    import torch

    from slm_training.models.grammar import dfa_admits_token, pick_constrained_token

    tok = _tok()
    prefix = tok.encode("root=", add_special=False)
    assert dfa_admits_token(tok, prefix, tok.token_to_id["="]) is False
    logits = torch.full((tok.vocab_size,), -20.0)
    logits[tok.token_to_id["="]] = 50.0
    logits[tok.token_to_id["Stack"]] = 1.0
    choice = pick_constrained_token(logits, tok, prefix, top_k=4)
    assert choice == tok.token_to_id["Stack"]


def test_pick_constrained_rejects_eos_on_incomplete_prefix() -> None:
    import torch

    from slm_training.models.grammar import pick_constrained_token

    tok = _tok()
    prefix = tok.encode("root", add_special=False)
    logits = torch.full((tok.vocab_size,), -20.0)
    logits[tok.eos_id] = 50.0
    logits[tok.token_to_id["="]] = 1.0
    choice = pick_constrained_token(logits, tok, prefix, top_k=4)
    assert choice != tok.eos_id


def test_ensure_valid_fallback_only_when_finalize() -> None:
    """Canned fallback must not silently inflate eval when finalize is off."""
    records = [
        ExampleRecord(
            id="t1",
            prompt="hero card",
            openui=SAMPLE,
            design_md="# Design\n",
            split="train",
            source="fixture",
        )
    ]
    cfg = TwoTowerConfig(
        context_backend="scratch",
        d_model=64,
        n_heads=2,
        context_layers=1,
        denoiser_layers=2,
        grammar_constrained=True,
        grammar_finalize_validate=False,
        grammar_ltr_max_tokens=16,
        max_target_len=32,
        max_prompt_len=32,
        seed=0,
    )
    model = TwoTowerModel.from_records(records, config=cfg, device="cpu")
    model.eval()
    ctx, ctx_pad = model._encode_context(["hero card"])
    # Force repair to fail so we exercise the post-repair branch.
    model._ltr_repair_from_bos = lambda *a, **k: "still-broken (("  # type: ignore[method-assign]
    raw = "not valid openui at all"
    out = model._ensure_valid_openui(raw, ctx, ctx_pad, 16, attempts=1)
    fallback = model._minimal_valid_openui()
    assert model._canonical_valid_openui(out) is None
    assert fallback is None or out != fallback

    model.config.grammar_finalize_validate = True
    try:
        certified = model._ensure_valid_openui(raw, ctx, ctx_pad, 16, attempts=1)
        assert model._canonical_valid_openui(certified) is not None
    except RuntimeError as exc:
        # Tiny fixture vocab may lack fallback templates — raise is still correct.
        assert "grammar_finalize_validate" in str(exc)


def test_admit_fill_rejects_hard_prefix() -> None:
    tok = _tok()
    eng = OpenUIIncrementalEngine()
    # Leading close-paren is never a valid OpenUI prefix.
    bad = [tok.bos_id, tok.token_to_id[")"], tok.mask_id, tok.eos_id]
    assert admit_fill(eng, tok, bad) is False


def test_draft_forced_ids_emits_equal() -> None:
    tok = _tok()
    eng = OpenUIIncrementalEngine()
    prefix = tok.encode("root", add_special=False)
    drafted = draft_forced_ids(eng, tok, list(prefix), max_tokens=2)
    assert drafted
    assert tok.id_to_token[drafted[0]] == "="


def test_train_fuse_and_cache_smoke() -> None:
    records = [
        ExampleRecord(
            id="t1",
            prompt="hero card",
            openui=SAMPLE,
            design_md="# Design\n",
            split="train",
            source="fixture",
        )
    ]
    cfg = TwoTowerConfig(
        context_backend="scratch",
        d_model=64,
        n_heads=2,
        context_layers=1,
        denoiser_layers=2,
        cache_context=True,
        fuse_ltr_loss=True,
        fastpath_aux_weight=0.1,
        grammar_fastpath=True,
        max_target_len=64,
        max_prompt_len=64,
        seed=0,
    )
    model = TwoTowerModel.from_records(records, config=cfg, device="cpu")
    loss1 = float(model.training_loss(records).detach())
    loss2 = float(model.training_loss(records).detach())
    assert loss1 == loss1 and loss2 == loss2  # finite
    assert "t1" in model._context_text_cache
    assert "t1" in model._target_ids_cache


def test_cactus_kernel_sketch_files_exist() -> None:
    from slm_training.bridge_utils import repo_root

    root = repo_root() / "src" / "slm_training" / "runtime" / "cactus" / "kernels"
    assert (root / "force_emit_sketch.hpp").is_file()
    assert (root / "maskgit_admit_sketch.hpp").is_file()
    assert (root / "README.md").is_file()
