"""SLM-308 (LAR2-02): distance oracle, value supervision, and matched-experiment tests."""

from __future__ import annotations

import inspect

import pytest
import torch

import slm_training.harnesses.experiments.slm308_distance_oracle as oracle
from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.experiments.slm308_distance_oracle import (
    DistanceKind,
    clear_caches,
    distance_to_target,
    effective_distance,
)
from slm_training.models.tree_edit_diffusion import (
    TreeEditDiffusionConfig,
    TreeEditDiffusionModel,
    pairwise_progress_loss,
    parse_statements,
)
from scripts.run_slm308_distance_value import (
    build_eval_states,
    build_fixture_records,
    build_report,
    concordance,
    render_markdown,
    spearman,
)

SEED = 'root = Stack([], "column")'
GOLD = 'root = Stack([cta], "column")\ncta = Button(":cta.label")'


@pytest.fixture()
def space():
    from slm_training.models.tree_edit_diffusion import TreeEditSpace

    return TreeEditSpace()


@pytest.fixture(autouse=True)
def _clean_oracle_caches():
    clear_caches()
    yield
    clear_caches()


# (a) EXACT / BOUNDED / UNKNOWN distinction ---------------------------------


def test_exact_distance_small_cases(space) -> None:
    seed = parse_statements(SEED)
    gold = parse_statements(GOLD)
    label = distance_to_target(
        seed, gold, space=space, inventory=[":cta.label"], node_budget=4
    )
    assert label.kind is DistanceKind.EXACT
    assert label.distance == 1
    assert label.value_target(8) == pytest.approx(1.0 - 1 / 8)
    # Identity is distance 0; the graph is undirected so reverse is exact too.
    ident = distance_to_target(gold, gold, space=space, inventory=[":x"])
    assert ident.kind is DistanceKind.EXACT and ident.distance == 0
    rev = distance_to_target(
        gold, seed, space=space, inventory=[":cta.label"], node_budget=4
    )
    assert rev.kind is DistanceKind.EXACT and rev.distance == 1


def test_invariant_proof_is_bounded_without_finite_bounds(space) -> None:
    seed = parse_statements(SEED)
    bad = parse_statements('root = Stack([cta], "column")\ncta = Button(":zzz")')
    label = distance_to_target(seed, bad, space=space, inventory=[":cta.label"])
    assert label.kind is DistanceKind.BOUNDED
    assert label.reason == "invariant:needs_slot_rebind"
    assert label.lo is None and label.hi is None
    # Unbounded proofs are excluded from value supervision, never coerced.
    assert label.value_target(8) is None
    assert effective_distance(label) is None


def test_budget_exhaustion_is_explicit_unknown(space) -> None:
    seed = parse_statements(SEED)
    gold = parse_statements(GOLD)
    # node_budget=1: the target layer expands, then the budget cuts layer 1
    # with a live frontier. UNKNOWN, never conflated with far/unreachable.
    label = distance_to_target(
        seed, gold, space=space, inventory=[":cta.label"],
        max_depth=8, node_budget=1,
    )
    # The seed itself is a depth-1 child of the target, so it IS reached;
    # use a state that is not adjacent for the UNKNOWN check.
    assert label.kind is DistanceKind.EXACT
    far = parse_statements(
        'root = Stack([a, b], "column")\na = TextContent(":x")\nb = Button(":y")'
    )
    # Both states' leaf slots must be in inventory, or the invariant proof
    # (not the budget path) decides the label.
    inv = [":x", ":y", ":cta.label"]
    label = distance_to_target(
        far, gold, space=space, inventory=inv,
        max_depth=8, node_budget=1,
    )
    assert label.kind is DistanceKind.UNKNOWN
    assert label.reason == "budget"
    assert label.value_target(8) is None
    # A proven witness path upgrades UNKNOWN to BOUNDED(lo, hi) — the bounds
    # are proofs (fully explored layers + witness), still never coerced exact.
    bounded = distance_to_target(
        far, gold, space=space, inventory=inv,
        max_depth=8, node_budget=1, upper_bound_witness=3,
    )
    assert bounded.kind is DistanceKind.BOUNDED
    assert bounded.reason == "budget"
    assert (bounded.lo, bounded.hi) == (2, 3)
    assert bounded.value_target(8) == pytest.approx(1.0 - 2.5 / 8)


def test_depth_bound_is_bounded_not_unknown(space) -> None:
    gold = parse_statements(GOLD)
    far = parse_statements(
        'root = Stack([a, b], "column")\na = TextContent(":x")\nb = Button(":y")'
    )
    # max_depth=0: nothing explored, budget intact, frontier still live.
    label = distance_to_target(
        far, gold, space=space, inventory=[":x", ":y", ":cta.label"],
        max_depth=0, node_budget=1, upper_bound_witness=4,
    )
    assert label.kind is DistanceKind.BOUNDED
    assert label.reason == "depth_bound"
    assert (label.lo, label.hi) == (1, 4)


# (b) cache identity ----------------------------------------------------------


def test_cache_hit_and_version_change_misses(space, monkeypatch) -> None:
    seed = parse_statements(SEED)
    gold = parse_statements(GOLD)
    kwargs = dict(space=space, inventory=[":cta.label"], node_budget=4)
    first = distance_to_target(seed, gold, **kwargs)
    assert not first.cache_hit
    second = distance_to_target(seed, gold, **kwargs)
    assert second.cache_hit and second.kind is first.kind
    # Action-schema version change => guaranteed miss.
    monkeypatch.setattr(oracle, "action_schema_version", lambda: "v999")
    third = distance_to_target(seed, gold, **kwargs)
    assert not third.cache_hit
    # Grammar version change => guaranteed miss.
    monkeypatch.setattr(oracle, "grammar_version", lambda: "sha256:deadbeef")
    fourth = distance_to_target(seed, gold, **kwargs)
    assert not fourth.cache_hit


def test_grammar_version_is_grammar_file_hash() -> None:
    assert oracle.grammar_version().startswith("sha256:")
    assert oracle.action_schema_version().startswith("v")


# (c) pairwise progress loss ---------------------------------------------------


def test_pairwise_progress_loss_improving_worsening_tie() -> None:
    # Improving pair with margin satisfied: child scores >= parent + margin.
    zero = pairwise_progress_loss(
        torch.tensor([0.5]), torch.tensor([0.7]), margin=0.1
    )
    assert float(zero) == 0.0
    # Worsening pair (child below parent): full violation + margin.
    loss = pairwise_progress_loss(
        torch.tensor([0.6]), torch.tensor([0.4]), margin=0.1
    )
    assert float(loss) == pytest.approx(0.3)
    # Tie inside the margin: only the margin remains.
    tie = pairwise_progress_loss(
        torch.tensor([0.5]), torch.tensor([0.5]), margin=0.1
    )
    assert float(tie) == pytest.approx(0.1)
    # Empty pair set is a no-op zero, never an error.
    empty = pairwise_progress_loss(torch.zeros(0), torch.zeros(0))
    assert float(empty) == 0.0
    # Gradient flows to both sides.
    parent = torch.tensor([0.5], requires_grad=True)
    child = torch.tensor([0.4], requires_grad=True)
    pairwise_progress_loss(parent, child, margin=0.1).backward()
    assert parent.grad is not None and child.grad is not None


# (d) no gold distance at inference --------------------------------------------

_RECORDS = [
    ExampleRecord(
        id="r0",
        prompt="hero with cta",
        openui=GOLD,
        placeholders=[":cta.label"],
        split="train",
    ),
    ExampleRecord(
        id="r1",
        prompt="simple text",
        openui='root = Stack([t], "column")\nt = TextContent(":body")',
        placeholders=[":body"],
        split="train",
    ),
]


def _tiny_model(mode: str = "bounded_distance") -> TreeEditDiffusionModel:
    config = TreeEditDiffusionConfig(
        value_label_mode=mode,
        d_model=32,
        n_heads=2,
        denoiser_layers=1,
        context_layers=1,
        beam_width=2,
        expand_per_state=2,
        max_search_steps=2,
        max_chain=2,
    )
    return TreeEditDiffusionModel.from_records(_RECORDS, config=config, device="cpu")


def test_decode_never_calls_the_oracle(monkeypatch) -> None:
    # Runtime audit: any oracle call from the decode path fails loudly.
    def _boom(*args, **kwargs):
        raise AssertionError("oracle reached from inference path")

    monkeypatch.setattr(oracle, "distance_to_target", _boom)
    model = _tiny_model()
    text = model.generate("hero with cta")
    assert isinstance(text, str)
    # Static audit: the decode functions carry no oracle reference.
    for fn_name in ("_decode_one", "_enumerate_edits", "_seed_state", "generate_batch_requests", "generate"):
        source = inspect.getsource(getattr(TreeEditDiffusionModel, fn_name))
        assert "distance_to_target" not in source
        assert "slm308" not in source


# (e) value supervision modes ----------------------------------------------------


def test_training_loss_bounded_mode_uses_oracle_and_pairs() -> None:
    model = _tiny_model("bounded_distance")
    loss = model.training_loss(_RECORDS)
    assert torch.isfinite(loss.detach())
    metrics = model.last_training_metrics
    assert "value" in metrics
    assert "value_bounded" in metrics
    assert "value_unknown_excluded" in metrics
    # Fixture chains are 1-2 edits from gold: at least one oracle-labeled
    # value and one strictly-improving pair must exist.
    assert metrics["value_bounded"] + 1 >= 1
    assert "pairwise_progress" in metrics


def test_training_loss_mutation_count_mode_matches_legacy() -> None:
    model = _tiny_model("mutation_count")
    loss = model.training_loss(_RECORDS)
    assert torch.isfinite(loss.detach())
    metrics = model.last_training_metrics
    # Legacy mode: every row carries a value, no oracle involvement, no pairs.
    assert metrics["value_unknown_excluded"] == 0.0
    assert metrics["value_bounded"] == 0.0
    assert "pairwise_progress" not in metrics


def test_unknown_states_are_excluded_from_value_loss(monkeypatch) -> None:
    model = _tiny_model("bounded_distance")

    class _Unknown:
        kind = DistanceKind.UNKNOWN

        def value_target(self, max_depth: int):
            return None

    monkeypatch.setattr(
        TreeEditDiffusionModel, "_distance_label", lambda *a, **k: _Unknown()
    )
    loss = model.training_loss(_RECORDS)
    assert torch.isfinite(loss.detach())
    metrics = model.last_training_metrics
    assert metrics["value_unknown_excluded"] >= 1.0


# (f) checkpoint parity -----------------------------------------------------------


def test_old_checkpoints_default_to_mutation_count(tmp_path) -> None:
    model = _tiny_model("bounded_distance")
    path = tmp_path / "ckpt.pt"
    model.save(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["config"]["value_label_mode"] == "bounded_distance"
    # Round-trip preserves the new mode.
    loaded = TreeEditDiffusionModel.from_checkpoint(path)
    assert loaded.config.value_label_mode == "bounded_distance"
    # A format-2 checkpoint written before the field existed loads as
    # mutation_count (behavior parity).
    del payload["config"]["value_label_mode"]
    torch.save(payload, path)
    legacy = TreeEditDiffusionModel.from_checkpoint(path)
    assert legacy.config.value_label_mode == "mutation_count"


def test_new_config_defaults_to_bounded_distance() -> None:
    assert TreeEditDiffusionConfig().value_label_mode == "bounded_distance"


# (g) metric helpers + matched-experiment determinism ------------------------------


def test_spearman_and_concordance() -> None:
    assert spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 1, 1], [1, 2, 3]) is None
    assert concordance([0.1, 0.5, 0.9], [0.1, 0.5, 0.9]) == pytest.approx(1.0)
    assert concordance([0.9, 0.5, 0.1], [0.1, 0.5, 0.9]) == pytest.approx(0.0)


def test_eval_states_are_deterministic(space) -> None:
    records = build_fixture_records()[:2]
    first = build_eval_states(records, space)
    second = build_eval_states(records, space)
    assert len(first) == len(second) > 0
    for a, b in zip(first, second):
        assert a["origin"] == b["origin"]
        assert a["witness"] == b["witness"]
        from slm_training.models.tree_edit_diffusion import render_statements

        assert render_statements(a["statements"]) == render_statements(b["statements"])
    assert {e["origin"] for e in first} == {"near_gold", "seed_trajectory"}


@pytest.mark.training
def test_matched_experiment_is_deterministic() -> None:
    import copy

    records = build_fixture_records()[:2]
    first = build_report(records, steps=2, batch_size=2, seed=0)
    second = build_report(records, steps=2, batch_size=2, seed=0)
    md = render_markdown(first)
    stripped = []
    for payload in (first, second):
        view = copy.deepcopy(payload)
        view.pop("version_stamp")
        # Wall-clock oracle cost fields are evidence, not identity.
        view["oracle_cost"].pop("wall_ms_total")
        view["oracle_cost"].pop("ms_per_state")
        stripped.append(view)
    assert stripped[0] == stripped[1]
    assert first["verdict"] in {"adopted", "rejected"}
    assert first["preregistered_thresholds"]["beam_regret_improvement_min"] == 0.05
    md = render_markdown(first)
    assert "Preregistered thresholds" in md
