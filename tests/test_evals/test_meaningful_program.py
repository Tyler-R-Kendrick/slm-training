from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from slm_training.dsl.schema import ExampleRecord
from slm_training.evals.meaningful_program import (
    CheckStatus,
    _implementation_hash,
    aggregate_meaning_reports_v2,
    binding_aware_meaningful_v2,
)
from slm_training.harnesses.model_build.eval_runner import meaningful_program_v1

CORPUS = Path("src/slm_training/resources/evals/meaningful_v2_gaming.jsonl")


def _record(case: dict[str, object]) -> ExampleRecord:
    return ExampleRecord(
        id=str(case["id"]),
        prompt=str(case["prompt"]),
        openui=str(case["prediction"]),
        split="adversarial",
        source="deterministic",
    )


def test_binding_aware_v2_gaming_corpus() -> None:
    for line in CORPUS.read_text(encoding="utf-8").splitlines():
        case = json.loads(line)
        report = binding_aware_meaningful_v2(
            case["prediction"], record=_record(case)
        )
        assert report.verdict is case["expected_verdict"], case["id"]
        assert set(case["expected_reason_codes"]) <= set(report.reason_codes), case[
            "id"
        ]
        assert report.metric_implementation_hash
        assert report.to_dict()["coverage_known"] is True


def test_unknown_prompt_contract_never_scores_positive() -> None:
    record = ExampleRecord(
        id="unknown",
        prompt="Make something useful",
        openui='root = Button(":cta.label")',
    )
    report = binding_aware_meaningful_v2(record.openui, record=record)
    assert report.verdict is False
    assert any(check.status is CheckStatus.UNKNOWN for check in report.checks)


def test_hard_prompt_inventory_rejects_extra_placeholder_identity() -> None:
    record = ExampleRecord(
        id="extra",
        prompt="Build a Stack. Placeholders: :hero.title",
        openui=(
            'root = Stack([title, extra])\n'
            'title = TextContent(":hero.title")\n'
            'extra = TextContent(":hero.extra")'
        ),
    )
    report = binding_aware_meaningful_v2(record.openui, record=record)
    assert report.verdict is False
    assert "unexpected_placeholder_identity" in report.reason_codes


def test_component_only_contract_keeps_placeholder_coverage_unknown() -> None:
    record = ExampleRecord(
        id="component-only",
        prompt="Build a Button",
        openui='root = Button(":cta.label")',
    )
    report = binding_aware_meaningful_v2(record.openui, record=record)
    assert report.verdict is False
    assert "required_inventory_unknown" in report.reason_codes


def test_placeholder_only_contract_keeps_component_relevance_unknown() -> None:
    record = ExampleRecord(
        id="slot-only",
        prompt="Placeholders: :hero.title",
        openui='root = TextContent(":hero.title")',
    )
    report = binding_aware_meaningful_v2(record.openui, record=record)
    assert report.verdict is False
    assert "prompt_contract_unknown" in report.reason_codes


def test_v2_rejects_swapped_placeholder_semantic_roles() -> None:
    source = (
        'root = Stack([button, title])\n'
        'button = Button(":hero.title")\n'
        'title = TextContent(":actions.save")'
    )
    report = binding_aware_meaningful_v2(
        source,
        record=ExampleRecord(
            id="swapped",
            prompt=(
                "Build a Button and TextContent. "
                "Placeholders: :actions.save :hero.title"
            ),
            openui=source,
        ),
    )
    assert report.verdict is False
    assert "placeholder_semantic_role_mismatch" in report.reason_codes


def test_v2_accepts_schema_declared_modal_title_role() -> None:
    source = (
        'body = TextContent(":modal.body")\n'
        'confirm = Button(":modal.confirm")\n'
        'actions = Buttons([confirm])\n'
        'dialog = Modal(":modal.title", true, [body, actions])\n'
        'root = Stack([dialog], "column")'
    )
    report = binding_aware_meaningful_v2(
        source,
        record=ExampleRecord(
            id="modal-title",
            prompt=(
                "Build a Modal, TextContent, and Button. "
                "Placeholders: :modal.title :modal.body :modal.confirm"
            ),
            openui=source,
        ),
    )

    assert report.verdict is True


def test_v2_accepts_display_body_and_value_aliases_from_dashboard_gold() -> None:
    source = (
        'status = Callout("info", ":dash.status.title", ":dash.status.body")\n'
        'm1 = Card([TextContent(":dash.m1.value")])\n'
        'm2 = Card([TextContent(":dash.m2.value")])\n'
        'root = Stack([status, m1, m2], "column")'
    )
    report = binding_aware_meaningful_v2(
        source,
        record=ExampleRecord(
            id="dashboard-display-aliases",
            prompt=(
                "Build a status Callout and two Cards. "
                "Placeholders: :dash.status.title :dash.status.body "
                ":dash.m1.value :dash.m2.value"
            ),
            openui=source,
        ),
    )

    assert report.verdict is True


def test_v2_accepts_heading_and_kicker_as_display_text() -> None:
    source = (
        'kicker = TextContent(":hero.kicker")\n'
        'heading = TextContent(":callout.heading")\n'
        'root = Stack([kicker, heading], "column")'
    )
    report = binding_aware_meaningful_v2(
        source,
        record=ExampleRecord(
            id="display-text-aliases",
            prompt=(
                "Build a Stack with display text. "
                "Placeholders: :hero.kicker :callout.heading"
            ),
            openui=source,
        ),
    )

    assert report.verdict is True


def test_v2_recognizes_singular_prose_for_plural_schema_component() -> None:
    source = (
        'root = Tabs([TabItem("overview", ":tabs.trigger", '
        '[TextContent(":tabs.text")])])'
    )
    report = binding_aware_meaningful_v2(
        source,
        record=ExampleRecord(
            id="singular-tabs-prose",
            prompt=(
                "Build a two-tab panel. "
                "Placeholders: :tabs.trigger :tabs.text"
            ),
            openui=source,
        ),
    )

    assert report.verdict is True
    assert "prompt_contract_unknown" not in report.reason_codes


def test_v2_accepts_numbered_tab_triggers_and_overview_text() -> None:
    source = (
        'overview = TextContent(":tabs.overview")\n'
        'one = TabItem("one", ":tabs.tab1", [overview])\n'
        'two = TabItem("two", ":tabs.tab2", [overview])\n'
        'root = Tabs([one, two])'
    )
    report = binding_aware_meaningful_v2(
        source,
        record=ExampleRecord(
            id="numbered-tab-roles",
            prompt=(
                "Build a two-tab panel. Placeholders: "
                ":tabs.overview :tabs.tab1 :tabs.tab2"
            ),
            openui=source,
        ),
    )

    assert report.verdict is True


def test_v2_preserves_form_slots_in_input_placeholder_property() -> None:
    source = (
        'root = Stack([name, email], "column")\n'
        'name = Input("", ":auth.name")\n'
        'email = Input("", ":auth.email")'
    )
    report = binding_aware_meaningful_v2(
        source,
        record=ExampleRecord(
            id="auth-inputs",
            prompt="Build two Inputs. Placeholders: :auth.name :auth.email",
            openui=source,
        ),
    )

    assert report.verdict is True


@pytest.mark.parametrize(
    "source",
    [
        'items = Query("q", {}, {rows: []})\nroot = Button(@Count(items.rows))',
        'items = Query("q", {}, {rows: []})\nroot = Stack(@Count(items.rows))',
    ],
)
def test_v2_rejects_known_dynamic_schema_role_mismatch(source: str) -> None:
    report = binding_aware_meaningful_v2(
        source,
        record=ExampleRecord(
            id="dynamic-role",
            prompt="Build a Button. Placeholders: :cta.label",
            openui=source,
        ),
    )
    assert report.verdict is False
    assert any(reason.startswith("schema_value_role_mismatch:") for reason in report.reason_codes)


@pytest.mark.parametrize(
    "source",
    [
        'unused = Query("q", {}, {})\nroot = Button(":cta.label")',
        '$a = $b\n$b = $a\nroot = Button(":cta.label")',
    ],
)
def test_v2_rejects_unreachable_runtime_or_state_bindings(source: str) -> None:
    report = binding_aware_meaningful_v2(
        source,
        record=ExampleRecord(
            id="dead-runtime",
            prompt="Build a Button. Placeholders: :cta.label",
            openui=source,
        ),
    )
    assert "unreachable_binding" in report.reason_codes


def test_v2_typed_array_object_item_keys_are_not_treated_as_unresolved_refs() -> None:
    """Regression for E618: a regex-based Gate.REFERENCES fallback used to run
    for any non-runtime-syntax source and misread bare object-literal property
    keys (e.g. ``src:``/``alt:`` in a typed-array item like
    ``{src: ..., alt: ...}``) as unresolved variable references, permanently
    failing ``binding_correctness`` (reason ``reference_graph_invalid``) for
    any correctly produced typed-array-of-objects prediction -- independent of
    model quality. Confirmed live in real E612-E617 eval evidence
    (docs/design/iter-e612..e617*.json) before this fix."""
    source = (
        'root = Stack([v0], "column")\n'
        'v0 = ImageGallery([{src: ":hero.img", alt: ":hero.alt"}])'
    )
    report = binding_aware_meaningful_v2(
        source,
        record=ExampleRecord(
            id="typed-array-object-item",
            prompt="Image gallery. Placeholders: :hero.img :hero.alt",
            openui=source,
        ),
    )
    binding_check = next(
        check for check in report.checks if check.name == "binding_correctness"
    )
    assert binding_check.status is CheckStatus.PASS
    assert "reference_graph_invalid" not in report.reason_codes


def test_v2_resolves_placeholder_through_reachable_state_binding() -> None:
    source = '$label = ":cta.label"\nroot = Button($label)'
    report = binding_aware_meaningful_v2(
        source,
        record=ExampleRecord(
            id="state-slot",
            prompt="Build a Button. Placeholders: :cta.label",
            openui=source,
        ),
    )
    assert report.verdict is True


def test_negated_component_mention_is_not_a_hard_requirement() -> None:
    source = 'root = Card([title])\ntitle = TextContent(":hero.title")'
    report = binding_aware_meaningful_v2(
        source,
        record=ExampleRecord(
            id="negation",
            prompt="Build a Card, not a Button. Placeholders: :hero.title",
            openui=source,
        ),
    )
    assert "prompt_component_missing" not in report.reason_codes


@pytest.mark.parametrize(
    "prompt",
    [
        "Replace the Button with a Card. Placeholders: :hero.title",
        "Build a Card without a Button. Placeholders: :hero.title",
        "Build a Button-free Card. Placeholders: :hero.title",
    ],
)
def test_v2_excludes_replaced_or_negated_components(prompt: str) -> None:
    source = 'root = Card([title])\ntitle = TextContent(":hero.title")'
    report = binding_aware_meaningful_v2(
        source,
        record=ExampleRecord(id="replacement", prompt=prompt, openui=source),
    )
    assert report.prompt_contract.required_components == ("Card",)
    assert report.verdict is True


def test_v2_does_not_require_likeness_modifier() -> None:
    source = 'root = Stack([submit], "column")\nsubmit = Button(":form.submit")'
    report = binding_aware_meaningful_v2(
        source,
        record=ExampleRecord(
            id="likeness",
            prompt="Form-like stack with submit button. Placeholders: :form.submit",
            openui=source,
        ),
    )

    assert report.prompt_contract.required_components == ("Button", "Stack")
    assert report.verdict is True


def test_v2_enforces_explicit_component_multiplicity() -> None:
    source = 'root = Button(":cta.first")'
    report = binding_aware_meaningful_v2(
        source,
        record=ExampleRecord(
            id="multiplicity",
            prompt="Build two Buttons. Placeholders: :cta.first",
            openui=source,
        ),
    )
    assert report.prompt_contract.required_components == ("Button", "Button")
    assert report.verdict is False
    assert "required_component_missing" in report.reason_codes


def test_v2_rejects_input_placeholder_in_name_property() -> None:
    source = 'root = Input(":form.email.placeholder")'
    report = binding_aware_meaningful_v2(
        source,
        record=ExampleRecord(
            id="input-role-swap",
            prompt="Build an Input. Placeholders: :form.email.placeholder",
            openui=source,
        ),
    )
    assert report.verdict is False
    assert "placeholder_semantic_role_mismatch" in report.reason_codes
    row = report.placeholder_inventory[0]
    assert row["component"] == "Input"
    assert row["property"] == "name"


def test_v2_placeholder_spans_do_not_match_longer_identifiers() -> None:
    source = (
        'root = Stack([short, long])\n'
        'short = TextContent(":copy.title")\n'
        'long = TextContent(":copy.title.long")'
    )
    report = binding_aware_meaningful_v2(
        source,
        record=ExampleRecord(
            id="exact-spans",
            prompt=(
                "Build a Stack and TextContent. "
                "Placeholders: :copy.title :copy.title.long"
            ),
            openui=source,
        ),
    )
    rows = {row["placeholder"]: row for row in report.placeholder_inventory}
    short_span = rows[":copy.title"]["source_span"]
    long_span = rows[":copy.title.long"]["source_span"]
    assert source[slice(*short_span)] == ":copy.title"
    assert source[slice(*long_span)] == ":copy.title.long"


def test_v2_metric_hash_includes_semantic_dependencies() -> None:
    module = Path("src/slm_training/evals/meaningful_program.py").resolve()
    package = module.parents[1]
    dependencies = (
        module,
        package / "data" / "quality.py",
        package / "data" / "verify" / "stack.py",
        package / "dsl" / "parser.py",
        package / "dsl" / "placeholders.py",
    )
    digest = hashlib.sha256()
    for path in dependencies:
        digest.update(path.relative_to(package.parent).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    _implementation_hash.cache_clear()
    assert _implementation_hash() == digest.hexdigest()


def test_v2_fails_closed_for_unknown_runtime_result_type() -> None:
    source = 'items = Query("q", {}, {})\nroot = TextContent(items)'
    report = binding_aware_meaningful_v2(
        source,
        record=ExampleRecord(
            id="unknown-runtime-type",
            prompt="Build TextContent. Placeholders: :copy.title",
            openui=source,
        ),
    )
    assert report.verdict is False
    assert "schema_value_role_mismatch:TextContent.text" in report.reason_codes


def test_v2_is_alpha_rename_invariant() -> None:
    record = ExampleRecord(
        id="alpha",
        prompt="Build a Card. Placeholders: :hero.title",
        openui='root = Card([title])\ntitle = TextContent(":hero.title")',
    )
    renamed = 'root = Card([copy])\ncopy = TextContent(":hero.title")'
    left = binding_aware_meaningful_v2(record.openui, record=record)
    right = binding_aware_meaningful_v2(renamed, record=record)
    assert left.verdict is right.verdict is True
    assert [check.status for check in left.checks] == [
        check.status for check in right.checks
    ]


def test_v2_accepts_valid_runtime_bindings_and_dynamic_schema_values() -> None:
    source = (
        "root = Stack([button, count])\n"
        '$filter = "all"\n'
        'items = Query("get_items", {filter: $filter}, {rows: []})\n'
        'save = Mutation("save_item", {filter: $filter})\n'
        "submit = Action([@Run(save), @Run(items)])\n"
        'button = Button(":actions.save", submit)\n'
        'count = TextContent("" + @Count(items.rows))'
    )
    report = binding_aware_meaningful_v2(
        source,
        record=ExampleRecord(
            id="runtime",
            prompt=(
                "Build a Button and TextContent. "
                "Placeholders: :actions.save"
            ),
            openui=source,
        ),
    )
    assert report.verdict is True


def test_v2_aggregate_reports_strict_and_coverage_conditioned_rates() -> None:
    known = ExampleRecord(
        id="known",
        prompt="Build a Button. Placeholders: :cta.label",
        openui='root = Button(":cta.label")',
    )
    unknown = ExampleRecord(
        id="unknown",
        prompt="Build a useful interface",
        openui='root = Button(":cta.label")',
    )
    summary = aggregate_meaning_reports_v2(
        [
            binding_aware_meaningful_v2(known.openui, record=known),
            binding_aware_meaningful_v2(unknown.openui, record=unknown),
        ]
    )
    assert summary["strict_rate"] == 0.5
    assert summary["coverage_conditioned_rate"] == 1.0
    assert summary["coverage"] == 0.5


def test_design_md_example_placeholder_is_not_a_required_prompt_slot() -> None:
    record = ExampleRecord(
        id="soft-design-example",
        prompt="Build a hero card with title and body.",
        openui='root = Card([TextContent(":copy.title")])',
        design_md="For example, keep copy symbolic as `:hero.title`.",
        split="smoke",
        source="unit",
    )

    report = binding_aware_meaningful_v2(record.openui, record=record)

    assert report.prompt_contract.required_placeholders == ()
    assert "required_placeholder_missing" not in report.reason_codes


@pytest.mark.parametrize(
    ("source", "reason"),
    [
        ("root = Stack([])", "empty_root_stack"),
        ("root = Card([])", "empty_card"),
        ('root = Stack([x])\nx = TextContent(":x")', None),
        ("root = Separator()", "no_placeholders"),
    ],
)
def test_meaningful_program_v1_backward_lock(source: str, reason: str | None) -> None:
    ok, actual_reason, serialized = meaningful_program_v1(source)
    assert ok is (reason is None)
    assert actual_reason == reason
    assert serialized is not None


# E631: `_is_meaningful_program`'s literal "Card([])" substring check missed
# a component whose `children` array is genuinely empty but whose remaining
# (non-content) positional arguments got padded with unrelated values --
# exactly the shape `required_slot_margin_decode_weight` produced on
# `rico_eval_test_25`/`rico_eval_test_42`/`rico_eval_test_77` and on
# `ood_dashboard_01` (see docs/design iter-e629/e630). The fix walks the real
# parsed AST (`Program.root`) instead of the serialized text, so a padded
# empty `children` array is caught regardless of what else is stuffed into
# sibling props.
@pytest.mark.parametrize(
    ("source", "reason"),
    [
        # The exact rico_eval_test_25 shape (E629/E630): Card's children are
        # empty but its `variant` prop absorbed a stuffed slot.
        ('root = Card([], ":stuffed.variant")', "empty_card"),
        # The exact ood_dashboard_01 shape (E630): two stuffed non-content
        # args after an empty children array.
        (
            'root = Card([], ":ood.dash.m1.value", ":ood.dash.m1.value")',
            "empty_card",
        ),
        # Nested: nothing in the literal check's own substring form catches
        # this either, since the padding still breaks the exact "Card([])"
        # match even when nested inside a Stack.
        (
            'root = Stack([v0, v1], "column")\n'
            'v0 = Card([TextContent(":a")])\n'
            'v1 = Card([], ":stuffed.variant")',
            "empty_card",
        ),
        # Modal/Carousel also declare a required `children` array in the
        # schema but were never covered by the old literal check at all
        # (it only special-cased Stack/Card by name).
        ('root = Modal(":title", true, [])', "empty_children:Modal"),
        ('root = Carousel([])', "empty_children:Carousel"),
    ],
)
def test_meaningful_program_v1_catches_padded_empty_children(
    source: str, reason: str
) -> None:
    ok, actual_reason, serialized = meaningful_program_v1(source)
    assert ok is False
    assert actual_reason == reason
    assert serialized is not None


@pytest.mark.parametrize(
    "source",
    [
        # Non-empty children with other props also present must still pass
        # -- the fix must not reject a genuinely-populated component just
        # because it also carries non-content properties.
        'root = Card([TextContent(":a")], ":real.variant")',
        'root = Stack([Card([TextContent(":a")])], "column")',
    ],
)
def test_meaningful_program_v1_still_accepts_genuinely_nonempty_components(
    source: str,
) -> None:
    ok, reason, serialized = meaningful_program_v1(source)
    assert ok is True
    assert reason is None
    assert serialized is not None
