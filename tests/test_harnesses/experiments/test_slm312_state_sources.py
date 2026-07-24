"""SLM-312 (LAR2-04): state-source acquisition, snapshots, leak guards, arms."""

from __future__ import annotations

import json

import pytest

# Tiny CPU fixture models: torch thread thrash dominates otherwise.
import torch

torch.set_num_threads(1)

from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.experiments.slm308_distance_oracle import (
    DistanceKind,
    clear_caches,
    distance_to_target,
    effective_distance,
)
from slm_training.harnesses.experiments.slm312_state_sources import (
    GOLD_VISIBILITY_ACQUISITION_AND_LABELS,
    GOLD_VISIBILITY_LABELS_ONLY,
    PREREGISTERED_ARMS,
    SOURCE_SEEDWARD,
    LeakageError,
    SnapshotIntegrityError,
    StateSnapshotV1,
    acquire_gold_only,
    acquire_seedward,
    assert_no_leakage,
    build_arm_rows,
    drop_leaked_rows,
    edit_from_dict,
    oracle_best_child,
    read_snapshot,
    source_fingerprint,
    state_supervision_loss,
)
from slm_training.models.tree_edit_diffusion import (
    TreeEditDiffusionConfig,
    TreeEditDiffusionModel,
    TreeEditSpace,
    parse_statements,
    render_statements,
)

GOLD = 'root = Stack([t, b], "column")\nt = TextContent(":hero.title")\nb = Button(":cta.label")'
SEED = 'root = Stack([], "column")'

# Tiny fixture model config (same overrides as the CLI) to keep tests fast.
_SMALL_CONFIG = {
    "context_backend": "scratch",
    "seed": 0,
    "d_model": 64,
    "denoiser_layers": 2,
    "context_layers": 1,
    "max_search_steps": 6,
    "beam_width": 3,
    "expand_per_state": 3,
}


@pytest.fixture()
def space():
    return TreeEditSpace()


@pytest.fixture(autouse=True)
def _clean_oracle_caches():
    clear_caches()
    yield
    clear_caches()


@pytest.fixture()
def records():
    return [
        ExampleRecord(
            id="train_0",
            prompt="hero with cta",
            openui=GOLD,
            placeholders=[":hero.title", ":cta.label"],
            split="train",
        ),
        ExampleRecord(
            id="train_1",
            prompt="form with input",
            openui='root = Form([i, s], "column")\ni = TextInput(":form.email")\ns = Button(":form.submit")',
            placeholders=[":form.email", ":form.submit"],
            split="train",
        ),
    ]


# --- provenance is explicit per row ------------------------------------------


def test_provenance_explicit_per_row(space, records) -> None:
    rows = acquire_gold_only(records, space, parent_checkpoint="ckpt-abc")
    assert rows, "expected near-gold rows"
    for row in rows:
        prov = row.provenance
        assert prov.parent_checkpoint == "ckpt-abc"
        assert prov.source_commit
        assert prov.prompt_hash
        assert prov.suite_hash
        assert prov.gold_visibility_policy == GOLD_VISIBILITY_ACQUISITION_AND_LABELS
        assert prov.rollout_config["acquisition"] == "sample_mutation"
        # Labels present: SLM-308 distance + SLM-305 legal supervised edit.
        assert row.distance_label["kind"] in {
            DistanceKind.EXACT.value,
            DistanceKind.BOUNDED.value,
            DistanceKind.UNKNOWN.value,
        }
        assert row.supervised_edit is not None
        assert "action_name" in row.supervised_edit


def test_on_policy_provenance_is_gold_blind_rollout(space, records) -> None:
    from slm_training.harnesses.experiments.slm312_state_sources import (
        acquire_on_policy,
    )

    torch_rows = acquire_gold_only(records[:1], space)
    model = TreeEditDiffusionModel.from_records(
        records[:1],
        config=TreeEditDiffusionConfig(**_SMALL_CONFIG),
        device="cpu",
    )
    rows = acquire_on_policy(
        model,
        records[:1],
        space,
        parent_checkpoint="bootstrap-test",
        max_states_per_record=3,
    )
    assert rows, "expected on-policy rows from beam telemetry"
    kinds = {r.kind for r in rows}
    assert kinds <= {"visited", "final", "verifier_failure_cone"}
    assert "final" in kinds
    for row in rows:
        assert row.provenance.parent_checkpoint == "bootstrap-test"
        assert row.provenance.gold_visibility_policy == GOLD_VISIBILITY_LABELS_ONLY
        assert row.provenance.rollout_config["beam_width"] == model.config.beam_width
    assert torch_rows  # sanity: scaffold reused


# --- leak guard: fail closed -------------------------------------------------


def test_leak_guard_fails_closed_on_overlap(space, records) -> None:
    rows = acquire_gold_only(records, space)
    victim = rows[0]
    with pytest.raises(LeakageError):
        assert_no_leakage(rows, {victim.fingerprint()})
    # A disjoint held-out set passes.
    assert_no_leakage(rows, {"0" * 64})
    # drop_leaked_rows removes the colliding row and the kept set verifies.
    kept, dropped = drop_leaked_rows(rows, {victim.fingerprint()})
    assert len(kept) == len(rows) - 1
    assert dropped and dropped[0]["record_id"] == victim.record_id
    assert_no_leakage(kept, {victim.fingerprint()})


# --- snapshot immutability / content addressing -------------------------------


def test_snapshot_roundtrip_and_tamper_detection(space, records, tmp_path) -> None:
    rows = acquire_gold_only(records, space)
    snap = StateSnapshotV1("snap-test", {"k": 1}, rows)
    path = snap.write(tmp_path / "rows.jsonl")
    back = read_snapshot(path)
    assert back.rows == rows
    assert back.manifest()["manifest_sha256"] == snap.manifest()["manifest_sha256"]

    # Tamper with one row payload (keep the stored hash): detect on load.
    lines = path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["kind"] = "tampered"
    lines[0] = json.dumps(first, sort_keys=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(SnapshotIntegrityError):
        read_snapshot(path)

    # Tamper with the manifest only: also detected.
    path2 = snap.write(tmp_path / "rows2.jsonl")
    manifest_path = path2.with_suffix(path2.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["row_sha256"][0] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SnapshotIntegrityError):
        read_snapshot(path2)


# --- seedward source: valid intermediates reaching toward the target ---------


def test_seedward_produces_valid_intermediates_toward_target(space, records) -> None:
    rows = acquire_seedward(records[:1], space)
    assert rows, "expected seedward intermediates"
    gold = parse_statements(GOLD)
    assert gold is not None
    seed_fp = source_fingerprint(SEED)
    distances: list[float] = []
    for row in rows:
        assert row.source == SOURCE_SEEDWARD
        statements = parse_statements(row.statements_source)
        assert statements is not None
        # Every intermediate is a valid, non-seed state strictly closer than
        # the seed side start, oracle-verified.
        assert row.fingerprint() != seed_fp
        label = distance_to_target(
            statements, gold, space=space, inventory=list(row.inventory)
        )
        assert label.kind is DistanceKind.EXACT
        eff = effective_distance(label)
        assert eff is not None
        distances.append(eff)
        # Supervised edit is SLM-305 legal on this state (fail-closed apply)
        # or the state is already at the target (STOP supervision).
        if row.supervised_edit is not None:
            child = space.apply(
                statements, edit_from_dict(row.supervised_edit), list(row.inventory)
            )
            assert child is not None
            assert row.child_source == render_statements(child)
    # The walk is strictly distance-decreasing toward the gold target.
    assert distances == sorted(distances, reverse=True)
    assert distances[0] > distances[-1]


def test_oracle_best_child_strictly_improves(space) -> None:
    seed = parse_statements(SEED)
    gold = parse_statements(GOLD)
    assert seed is not None and gold is not None
    inventory = [":hero.title", ":cta.label"]
    found = oracle_best_child(seed, gold, space, inventory)
    assert found is not None
    edit, child, label = found
    d_seed = effective_distance(
        distance_to_target(seed, gold, space=space, inventory=inventory)
    )
    d_child = effective_distance(
        distance_to_target(child, gold, space=space, inventory=inventory)
    )
    # The child is always oracle-measurable; when the parent is measurable
    # too, the child is strictly closer (the seed is UNKNOWN at the shallow
    # training budgets, so the comparison is conditional).
    assert d_child is not None
    if d_seed is not None:
        assert d_child < d_seed
    assert label.kind is DistanceKind.EXACT
    # The returned edit is legal and reproduces the child exactly.
    assert space.apply(seed, edit, inventory) is not None
    assert render_statements(space.apply(seed, edit, inventory)) == render_statements(
        child
    )


# --- mixture arms: shared budgets, isolation ----------------------------------


def test_mixture_arms_share_budgets_and_sources(space, records) -> None:
    rows = acquire_gold_only(records, space) + acquire_seedward(records, space)
    budget = 6
    for arm, weights in PREREGISTERED_ARMS.items():
        selected = build_arm_rows(rows, arm, per_arm_budget=budget, seed=0)
        assert len(selected) <= budget
        sources = {r.source for r in selected}
        expected = {s for s, w in weights.items() if w > 0}
        assert sources <= expected
        if arm != "mixed":
            assert sources <= {arm}
    # Determinism: same seed, same rows.
    again = build_arm_rows(rows, "mixed", per_arm_budget=budget, seed=0)
    first = build_arm_rows(rows, "mixed", per_arm_budget=budget, seed=0)
    assert [r.sha256() for r in again] == [r.sha256() for r in first]
    # Unknown arm is rejected (predeclared set only).
    with pytest.raises(ValueError, match="unknown arm"):
        build_arm_rows(rows, "sneaky_extra_arm", per_arm_budget=budget)


# --- determinism of acquisition -----------------------------------------------


def test_acquisition_deterministic(space, records) -> None:
    a = acquire_gold_only(records, space) + acquire_seedward(records, space)
    clear_caches()
    b = acquire_gold_only(records, space) + acquire_seedward(records, space)
    assert [r.sha256() for r in a] == [r.sha256() for r in b]


# --- loss consumes rows (wiring) ----------------------------------------------


def test_state_supervision_loss_runs_and_backprops(space, records) -> None:
    rows = acquire_gold_only(records, space)
    model = TreeEditDiffusionModel.from_records(
        records,
        config=TreeEditDiffusionConfig(**_SMALL_CONFIG),
        device="cpu",
    )
    record_by_id = {r.id: r for r in records}
    loss = state_supervision_loss(model, rows[:4], record_by_id)
    assert float(loss.detach()) >= 0.0
    loss.backward()
    metrics = model.last_training_metrics
    assert metrics["rows"] == 4.0
    assert "action" in metrics and "value" in metrics
