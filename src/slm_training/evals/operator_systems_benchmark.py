"""DSH3-32 (SLM-407): operator-inference systems benchmark.

Defines ``OperatorServingWorkV1``, one systems numeraire for comparing
operator-inference arms, and runs it over small fixture-scale operator states.

Scope (read before trusting a number here)
--------------------------------------------
* This is a **fixture-scale wiring harness**, like ``evals/tree_edit_scaling.py``
  and ``evals/cap2_operator.py``. It loads no trained checkpoint and makes no
  quality or ship claim; see ``docs/design/dsh3-32-operator-systems-benchmark
  -20260726.md`` for the full disposition and honest caveats.
* Five arms are measured for real: ``compiler_only_forced``,
  ``serialized_actions``, ``typed_policy_singleton_bypass_on``,
  ``typed_policy_singleton_bypass_off``, and ``x22_tree_edit``. The issue text
  also names a ``full_generation`` arm (plain end-to-end causal-lm decode with
  no operator legal set at all); wiring that honestly needs a trained/
  checkpointed causal-lm decode path that does not exist at fixture scale, so
  it is left in ``UNRUN_ARMS`` rather than fabricated.
* ``dry_runs`` counts real ``OperatorLibraryV1.dry_run`` calls (via
  ``OperatorLegalEntryV1.evaluated_combinations``). ``OperatorLibraryV1`` does
  not separate an "executor" call from a "validator" call at this layer, so
  ``executor_calls``/``validator_calls`` are only populated where a
  ``dsl.parser.validate`` call is actually observable (the X22 arm); elsewhere
  they are left at 0 rather than assumed equal to ``dry_runs``.
* Only the legal-set/reference-table cache axis is measured (``cache_state``
  "cold" vs "warm": whether legal-set enumeration is redone). Prompt/context
  caching, batching, and prefix sharing are named in the issue but not
  implemented by any arm in this repository today, so they are not measured
  here -- an explicit gap, not a fabricated zero.
"""

from __future__ import annotations

import hashlib
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Iterator, Literal

from slm_training.dsl.operators import (
    ActionEffectV1,
    ApplicationProvenanceV1,
    AstOperatorV1,
    BindingPhase,
    CompilerCoverage,
    LegalSetCoverage,
    OperatorArgumentSlotV1,
    OperatorLegalSetV1,
    OperatorLibraryV1,
    OperatorMutationV1,
    OperatorRejectedError,
    OperatorStateV1,
    RefKind,
    ReferenceDescriptorV1,
    ReferenceTableV1,
    RegisteredOperatorV1,
    build_reference_table,
    enumerate_operator_legal_set,
)
from slm_training.dsl.pack import DslPack, get_pack
from slm_training.harnesses.model_build.plugin import GenerationRequest
from slm_training.harnesses.experiments.typed_operator_policy import (
    TypedOperatorPolicyExampleV1,
    TypedOperatorPolicyScorer,
    decide_typed_operator_policy,
)
from slm_training.models.operator_policy_view import build_operator_policy_input
from slm_training.models.tokenizer import OpenUITokenizer
from slm_training.models.tree_edit_diffusion import (
    TreeEditDiffusionConfig,
    TreeEditDiffusionModel,
)
from slm_training.versioning import UNKNOWN, build_version_stamp

OPERATOR_SERVING_WORK_SCHEMA = "operator_serving_work/v1"

ArmId = Literal[
    "compiler_only_forced",
    "serialized_actions",
    "typed_policy_singleton_bypass_on",
    "typed_policy_singleton_bypass_off",
    "x22_tree_edit",
]

OPERATOR_SERVING_ARMS: tuple[ArmId, ...] = (
    "compiler_only_forced",
    "serialized_actions",
    "typed_policy_singleton_bypass_on",
    "typed_policy_singleton_bypass_off",
    "x22_tree_edit",
)

# The issue's own acceptance text names a sixth arm ("full generation": plain
# end-to-end causal-lm decode with no operator legal set). See module
# docstring -- explicitly unrun, never fabricated.
UNRUN_ARMS: tuple[str, ...] = ("full_generation",)

CacheState = Literal["cold", "warm", "n/a"]


@dataclass(frozen=True)
class OperatorServingWorkV1:
    """One systems-numeraire measurement for one arm on one fixture state."""

    arm_id: str
    stratum: str
    cache_state: CacheState
    coverage: str
    legal_set_size: int
    operator_row_count: int
    actions_committed: int
    model_forwards: int
    policy_rows_scored: int
    dry_runs: int
    executor_calls: int
    validator_calls: int
    wall_ms: float
    cpu_ms: float
    failed: bool
    schema: str = OPERATOR_SERVING_WORK_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "legal_set_size",
            "operator_row_count",
            "actions_committed",
            "model_forwards",
            "policy_rows_scored",
            "dry_runs",
            "executor_calls",
            "validator_calls",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"OperatorServingWorkV1.{name} must be non-negative")
        if self.wall_ms < 0.0 or self.cpu_ms < 0.0:
            raise ValueError("OperatorServingWorkV1 timings must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OperatorServingBenchmarkV1:
    """Fixture-scale disposition report for the DSH3-32 systems numeraire."""

    arms_measured: tuple[str, ...]
    arms_unrun: tuple[str, ...]
    rows: tuple[OperatorServingWorkV1, ...]
    claim_class: str = "wiring"
    ship_eligible: bool = False
    version_stamp: dict[str, Any] = field(default_factory=dict)
    schema: str = "operator_serving_benchmark/v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "arms_measured": list(self.arms_measured),
            "arms_unrun": list(self.arms_unrun),
            "claim_class": self.claim_class,
            "ship_eligible": self.ship_eligible,
            "rows": [row.to_dict() for row in self.rows],
            "version_stamp": self.version_stamp,
        }


@contextmanager
def _measure() -> Iterator[dict[str, float]]:
    box = {"wall_ms": 0.0, "cpu_ms": 0.0}
    wall0 = time.perf_counter()
    cpu0 = time.process_time()
    try:
        yield box
    finally:
        box["wall_ms"] = (time.perf_counter() - wall0) * 1000.0
        box["cpu_ms"] = (time.process_time() - cpu0) * 1000.0


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


_SOURCE = 'root = TextContent(":hero.title")'


def _descriptor(index: int) -> ReferenceDescriptorV1:
    return ReferenceDescriptorV1(
        ref_kind=RefKind.VALUE,
        semantic_fingerprint=_sha(f"operator-systems-benchmark-value-{index}"),
        value_type="openui.string",
    )


@dataclass(frozen=True)
class OperatorFixtureV1:
    """One reusable (pack, library, state, table, provenance) tuple."""

    stratum: str
    pack: DslPack
    library: OperatorLibraryV1
    state: OperatorStateV1
    table: ReferenceTableV1
    provenance: ApplicationProvenanceV1


def build_operator_fixture(
    *,
    stratum: str,
    operator_count: int,
    descriptor_count: int,
    allowed_indices: frozenset[int],
    repeated: bool = False,
    seed: int = 11,
) -> OperatorFixtureV1:
    """Build a small real operator legal-set fixture for one benchmark stratum."""
    base_pack = get_pack("openui")
    state = OperatorStateV1.from_source(base_pack, _SOURCE)
    descriptors = tuple(_descriptor(index) for index in range(descriptor_count))
    table = build_reference_table(
        request_id="operator-systems-benchmark",
        state_digest=state.state_digest,
        branch_digest=_sha(f"operator-systems-benchmark-branch-{stratum}"),
        descriptors=descriptors,
        seed=seed,
    )
    semantic_by_ref = {
        entry.ref: entry.descriptor.semantic_fingerprint for entry in table.entries
    }
    allowed = {_descriptor(index).semantic_fingerprint for index in allowed_indices}

    def _make_execute(tag: str):
        def execute(operator_state, arguments):
            semantic = semantic_by_ref[arguments[0].value]
            if semantic not in allowed:
                raise OperatorRejectedError("operator_systems_benchmark.not_legal")
            return OperatorMutationV1(
                source=operator_state.source.replace(":hero.title", f":hero.{tag}"),
                effect=ActionEffectV1(compiler_coverage=CompilerCoverage.EXACT),
            )

        return execute

    registered = []
    for op_index in range(operator_count):
        declaration = AstOperatorV1(
            operator_id=f"openui.benchmark_op_{op_index}",
            version="v1",
            domain="openui.ast",
            codomain="openui.ast",
            argument_slots=(
                OperatorArgumentSlotV1(
                    "value", RefKind.VALUE, BindingPhase.APPLICATION, repeated=repeated
                ),
            ),
            preconditions=(),
            effect_signature=(),
            locality="node",
            cost=1.0,
        )
        registered.append(
            RegisteredOperatorV1(declaration, _make_execute(f"body_{op_index}"))
        )
    library = OperatorLibraryV1(tuple(registered))
    pack = replace(base_pack, operator_library=library)
    provenance = ApplicationProvenanceV1(
        pack_id="openui",
        compiler_id="openui.operator_systems_benchmark",
        compiler_version="v1",
        source_artifact_digest=_sha(_SOURCE),
        request_id="operator-systems-benchmark",
    )
    return OperatorFixtureV1(
        stratum=stratum, pack=pack, library=library, state=state, table=table,
        provenance=provenance,
    )


def default_operator_fixtures() -> tuple[OperatorFixtureV1, ...]:
    """Three strata: exactly-one-legal-operator, two-legal-operators, PARTIAL."""
    return (
        build_operator_fixture(
            stratum="singleton",
            operator_count=1,
            descriptor_count=3,
            allowed_indices=frozenset({0}),
        ),
        build_operator_fixture(
            stratum="ambiguous",
            operator_count=2,
            descriptor_count=3,
            allowed_indices=frozenset({0, 1}),
        ),
        build_operator_fixture(
            stratum="partial",
            operator_count=1,
            descriptor_count=3,
            allowed_indices=frozenset({0}),
            repeated=True,
        ),
    )


def _enumerate(fixture: OperatorFixtureV1) -> OperatorLegalSetV1:
    return enumerate_operator_legal_set(
        pack=fixture.pack,
        library=fixture.library,
        state=fixture.state,
        reference_table=fixture.table,
        provenance=fixture.provenance,
    )


def _dry_runs(legal_set: OperatorLegalSetV1) -> int:
    return sum(entry.evaluated_combinations for entry in legal_set.entries)


def run_compiler_only_forced(
    fixture: OperatorFixtureV1, *, cache_state: CacheState, cached: OperatorLegalSetV1 | None
) -> tuple[OperatorLegalSetV1, OperatorServingWorkV1]:
    """Deterministic compiler enumeration + first-in-order pick, zero forwards."""
    with _measure() as box:
        legal_set = cached if cached is not None else _enumerate(fixture)
        actions = legal_set.all_serialized_actions
        chosen = actions[0] if actions else None
    dry_runs = 0 if cached is not None else _dry_runs(legal_set)
    row = OperatorServingWorkV1(
        arm_id="compiler_only_forced",
        stratum=fixture.stratum,
        cache_state=cache_state,
        coverage=legal_set.coverage.value,
        legal_set_size=len(actions),
        operator_row_count=len(legal_set.entries),
        actions_committed=1 if chosen is not None else 0,
        model_forwards=0,
        policy_rows_scored=0,
        dry_runs=dry_runs,
        executor_calls=0,
        validator_calls=0,
        wall_ms=box["wall_ms"],
        cpu_ms=box["cpu_ms"],
        failed=chosen is None,
    )
    return legal_set, row


def run_serialized_actions(
    fixture: OperatorFixtureV1, *, cache_state: CacheState, cached: OperatorLegalSetV1 | None
) -> tuple[OperatorLegalSetV1, OperatorServingWorkV1]:
    """Enumerate and emit the full candidate wire format; nothing committed."""
    with _measure() as box:
        legal_set = cached if cached is not None else _enumerate(fixture)
        actions = legal_set.all_serialized_actions
    dry_runs = 0 if cached is not None else _dry_runs(legal_set)
    row = OperatorServingWorkV1(
        arm_id="serialized_actions",
        stratum=fixture.stratum,
        cache_state=cache_state,
        coverage=legal_set.coverage.value,
        legal_set_size=len(actions),
        operator_row_count=len(legal_set.entries),
        actions_committed=0,
        model_forwards=0,
        policy_rows_scored=0,
        dry_runs=dry_runs,
        executor_calls=0,
        validator_calls=0,
        wall_ms=box["wall_ms"],
        cpu_ms=box["cpu_ms"],
        failed=False,
    )
    return legal_set, row


def _typed_policy_example(
    fixture: OperatorFixtureV1, legal_set: OperatorLegalSetV1
) -> TypedOperatorPolicyExampleV1:
    view = build_operator_policy_input(fixture.table, legal_set, fixture.library)
    return TypedOperatorPolicyExampleV1(
        row_id=f"{fixture.stratum}-benchmark",
        view=view,
        accepted_action_row=0,
        accepted_argument_rows=(),
    )


def run_typed_policy(
    fixture: OperatorFixtureV1,
    *,
    cache_state: CacheState,
    cached: OperatorLegalSetV1 | None,
    bypass: bool,
) -> tuple[OperatorLegalSetV1, OperatorServingWorkV1]:
    """Typed-policy decision with the COMPLETE_SINGLETON bypass on or off.

    ``bypass=True`` uses ``decide_typed_operator_policy`` unmodified, so a
    true single-legal-operator state costs zero forwards. ``bypass=False``
    forces the scorer forward for every COMPLETE state (including true
    singletons) to measure the forward this repo's non-negotiable
    "deterministic/singleton bypass outranks any learned score" invariant is
    saving in practice.
    """
    with _measure() as box:
        legal_set = cached if cached is not None else _enumerate(fixture)
        example = _typed_policy_example(fixture, legal_set)
        scorer = TypedOperatorPolicyScorer.from_examples([example])
        if bypass or legal_set.coverage is not LegalSetCoverage.COMPLETE:
            # bypass=True always respects the real singleton-bypass routing;
            # PARTIAL coverage never forces regardless of the bypass toggle.
            decision = decide_typed_operator_policy(scorer, example)
            model_forwards = decision.model_forwards
            selected = decision.selected_action_row
        else:
            # bypass=False: force the scorer forward for every COMPLETE
            # state, including true singletons, to measure what the
            # singleton-bypass invariant is saving.
            action_logits, _ = scorer(example.view)
            selected = int(action_logits.argmax().item())
            model_forwards = 1
    dry_runs = 0 if cached is not None else _dry_runs(legal_set)
    row = OperatorServingWorkV1(
        arm_id=(
            "typed_policy_singleton_bypass_on"
            if bypass
            else "typed_policy_singleton_bypass_off"
        ),
        stratum=fixture.stratum,
        cache_state=cache_state,
        coverage=legal_set.coverage.value,
        legal_set_size=len(legal_set.all_serialized_actions),
        operator_row_count=len(legal_set.entries),
        actions_committed=1 if selected is not None else 0,
        model_forwards=model_forwards,
        policy_rows_scored=len(example.view.action_rows) if model_forwards else 0,
        dry_runs=dry_runs,
        executor_calls=0,
        validator_calls=0,
        wall_ms=box["wall_ms"],
        cpu_ms=box["cpu_ms"],
        failed=selected is None and legal_set.coverage is LegalSetCoverage.COMPLETE,
    )
    return legal_set, row


_X22_PROMPT = "Show a hero title."
_X22_INVENTORY: tuple[str, ...] = (":slot_0",)
# Matches the seed program TreeEditDiffusionModel._seed_state synthesizes for
# this inventory, so the tokenizer vocabulary covers every token the search
# can actually emit.
_X22_PROGRAM = 'root = Stack([n0], "column")\nn0 = TextContent(":slot_0")'


def _build_x22_model() -> TreeEditDiffusionModel:
    tokenizer = OpenUITokenizer.build([_X22_PROGRAM])
    config = TreeEditDiffusionConfig(
        d_model=16, n_heads=2, context_layers=1, denoiser_layers=1,
        max_search_steps=4, beam_width=2, expand_per_state=2,
    )
    return TreeEditDiffusionModel(tokenizer, config=config, device="cpu")


def run_x22_tree_edit(model: TreeEditDiffusionModel) -> OperatorServingWorkV1:
    """One X22 decode; forwards/rows come from the model's own evidence dict."""
    from slm_training.dsl.parser import validate

    with _measure() as box:
        outputs = model.generate_batch_requests(
            [GenerationRequest(prompt=_X22_PROMPT, slot_contract=_X22_INVENTORY)]
        )
        evidence = model.consume_generation_evidence()[0]
        try:
            validate(outputs[0])
            validator_calls = 1
            failed = False
        except Exception:  # noqa: BLE001
            validator_calls = 1
            failed = True
    return OperatorServingWorkV1(
        arm_id="x22_tree_edit",
        stratum="tree_edit_search",
        cache_state="n/a",
        coverage="n/a",
        legal_set_size=0,
        operator_row_count=0,
        actions_committed=0 if failed else 1,
        model_forwards=int(evidence.get("forwards", 0)),
        policy_rows_scored=int(evidence.get("forward_rows", 0)),
        dry_runs=0,
        executor_calls=0,
        validator_calls=validator_calls,
        wall_ms=box["wall_ms"],
        cpu_ms=box["cpu_ms"],
        failed=failed,
    )


STAMP_COMPONENTS: tuple[str, ...] = ("evals.operator_systems_benchmark",)


def run_operator_systems_benchmark(*, repeats: int = 2) -> OperatorServingBenchmarkV1:
    """Run every measured arm over every fixture stratum, cold then warm."""
    if repeats < 2:
        raise ValueError("repeats must be at least 2 to observe cold vs warm")
    rows: list[OperatorServingWorkV1] = []
    for fixture in default_operator_fixtures():
        compiler_cache: OperatorLegalSetV1 | None = None
        serialized_cache: OperatorLegalSetV1 | None = None
        bypass_on_cache: OperatorLegalSetV1 | None = None
        bypass_off_cache: OperatorLegalSetV1 | None = None
        for attempt in range(repeats):
            cache_state: CacheState = "cold" if attempt == 0 else "warm"
            compiler_cache, row = run_compiler_only_forced(
                fixture, cache_state=cache_state, cached=compiler_cache
            )
            rows.append(row)
            serialized_cache, row = run_serialized_actions(
                fixture, cache_state=cache_state, cached=serialized_cache
            )
            rows.append(row)
            bypass_on_cache, row = run_typed_policy(
                fixture, cache_state=cache_state, cached=bypass_on_cache, bypass=True
            )
            rows.append(row)
            bypass_off_cache, row = run_typed_policy(
                fixture, cache_state=cache_state, cached=bypass_off_cache, bypass=False
            )
            rows.append(row)
    x22_model = _build_x22_model()
    for _ in range(repeats):
        rows.append(run_x22_tree_edit(x22_model))
    try:
        version_stamp = build_version_stamp(*STAMP_COMPONENTS)
    except KeyError:
        version_stamp = {
            "stamp_schema": UNKNOWN,
            "components": {cid: UNKNOWN for cid in STAMP_COMPONENTS},
            "note": "version stamp unavailable",
        }
    return OperatorServingBenchmarkV1(
        arms_measured=OPERATOR_SERVING_ARMS,
        arms_unrun=UNRUN_ARMS,
        rows=tuple(rows),
        version_stamp=version_stamp,
    )


__all__ = [
    "ArmId",
    "OPERATOR_SERVING_ARMS",
    "OPERATOR_SERVING_WORK_SCHEMA",
    "OperatorFixtureV1",
    "OperatorServingBenchmarkV1",
    "OperatorServingWorkV1",
    "UNRUN_ARMS",
    "build_operator_fixture",
    "default_operator_fixtures",
    "run_compiler_only_forced",
    "run_operator_systems_benchmark",
    "run_serialized_actions",
    "run_typed_policy",
    "run_x22_tree_edit",
]
