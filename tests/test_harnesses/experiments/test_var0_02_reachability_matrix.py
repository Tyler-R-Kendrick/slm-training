"""VAR0-02 (SLM-423): per-variant x suite edit-reachability matrix.

Two things this suite must prove:

1. Adding the ``variant`` parameter to ``analyze_reachability`` is a pure
   refactor -- every existing caller that omits it keeps computing exactly
   the same ``ReachabilityCase`` it always computed, bit-for-bit (the
   regression guard the issue's acceptance criteria call for). Corpora on
   disk have drifted since ``iter-slm305-edit-language-20260724.md`` was
   generated (rico grew from 6 to 35 records; the train corpus is a
   gitignored generated artifact and is not present here at all -- see
   ``scripts/run_var1_01_set_property_probe.py``'s own note on this same
   drift for VAR1-01), so this suite freezes the still-unchanged smoke /
   held_out / ood suite records as literals rather than reading
   ``test_seeds.jsonl`` at test time, and compares against the archived
   report's committed numbers for exactly those three suites.
2. ``build_matrix`` populates the ``tree_edit_diffusion`` row (measured or
   corpus_unavailable, never blank, never inferred) and reports every other
   registered variant as ``NOT_MEASURED`` with a real, distinct reason --
   never inferred from the tree-edit row's numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.dsl.variants import VARIANTS
from slm_training.harnesses.experiments.slm299_edit_reachability import (
    DEFAULT_SEED_SOURCE,
    TREE_EDIT_VARIANT,
    analyze_reachability,
)
from scripts.run_slm299_reachability_audit import build_report
from scripts.run_var0_02_reachability_matrix import (
    NOT_MEASURED_REASONS,
    STATUS_CORPUS_UNAVAILABLE,
    STATUS_MEASURED,
    STATUS_NOT_MEASURED,
    SUITES,
    build_matrix,
    render_markdown,
)

SEED = DEFAULT_SEED_SOURCE

_ARCHIVED_DOC = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "design"
    / "iter-slm305-edit-language-20260724.json"
)

# Frozen copies of the smoke / held_out / ood test_seeds.jsonl records as they
# stood when iter-slm305-edit-language-20260724.md was generated (and as they
# still stand today -- verified against the live file while writing this
# test). Embedded as literals, not read from disk, so this regression guard
# never breaks merely because test_seeds.jsonl grows a new record later (the
# same drift that already hit `rico` and `train` -- see the module docstring
# and scripts/run_var1_01_set_property_probe.py's own note on it).
_FROZEN_SMOKE_HELD_OUT_OOD: dict[str, list[dict[str, object]]] = {
    # build_report/build_matrix key off whichever suites the corpora dict
    # supplies; the empty lists below make train/adversarial/rico explicit
    # (matching load_corpora()'s always-six-keys shape) so both scripts
    # report them corpus_unavailable rather than simply omitting the key.
    "train": [],
    "adversarial": [],
    "rico": [],
    "smoke": [
        {
            "id": "smoke_hero_01",
            "openui": (
                'root = Stack([title, rule, hero], "column")\n'
                'title = TextContent(":smoke.hero.kicker")\n'
                "rule = Separator()\n"
                'head = CardHeader(":smoke.hero.title", ":smoke.hero.subtitle")\n'
                'body = TextContent(":smoke.hero.body")\n'
                "hero = Card([head, body])"
            ),
            "placeholders": [
                ":smoke.hero.kicker",
                ":smoke.hero.title",
                ":smoke.hero.subtitle",
                ":smoke.hero.body",
            ],
        },
        {
            "id": "smoke_button_01",
            "openui": (
                'root = Stack([row], "column")\n'
                'cta = Button(":smoke.cta.label")\n'
                "row = Buttons([cta])"
            ),
            "placeholders": [":smoke.cta.label"],
        },
        {
            "id": "smoke_callout_01",
            "openui": (
                'root = Stack([title, note], "column")\n'
                'title = TextContent(":smoke.callout.heading")\n'
                'note = Callout("info", ":smoke.callout.title", ":smoke.callout.body")'
            ),
            "placeholders": [
                ":smoke.callout.heading",
                ":smoke.callout.title",
                ":smoke.callout.body",
            ],
        },
    ],
    "held_out": [
        {
            "id": "held_out_form_01",
            "openui": (
                'root = Stack([title, note, form], "column")\n'
                'title = TextContent(":held.form.title")\n'
                'email = Input("$0", ":held.form.email", "email")\n'
                'field = FormControl(":held.form.email.label", email)\n'
                'note = Callout("info", ":held.form.hint.title", ":held.form.hint.body")\n'
                'submit = Button(":held.form.submit")\n'
                "actions = Buttons([submit])\n"
                'form = Form("$1", actions, [field])'
            ),
            "placeholders": [
                ":held.form.title",
                ":held.form.email",
                ":held.form.email.label",
                ":held.form.hint.title",
                ":held.form.hint.body",
                ":held.form.submit",
            ],
        },
        {
            "id": "held_out_dual_card_01",
            "openui": (
                'root = Stack([header, rule, row], "column")\n'
                'header = TextContent(":held.col.header")\n'
                "rule = Separator()\n"
                'left_title = TextContent(":held.col.left.title")\n'
                'left_body = TextContent(":held.col.left.body")\n'
                "left = Card([left_title, left_body])\n"
                'right_title = TextContent(":held.col.right.title")\n'
                'right_body = TextContent(":held.col.right.body")\n'
                "right = Card([right_title, right_body])\n"
                'row = Stack([left, right], "row")'
            ),
            "placeholders": [
                ":held.col.header",
                ":held.col.left.title",
                ":held.col.left.body",
                ":held.col.right.title",
                ":held.col.right.body",
            ],
        },
        {
            "id": "held_out_input_01",
            "openui": (
                'root = Stack([email, continueBtn], "column")\n'
                'email = Input("$0", ":held.login.email.placeholder", "email")\n'
                'continueBtn = Button(":held.login.continue")'
            ),
            "placeholders": [
                ":held.login.email.placeholder",
                ":held.login.continue",
            ],
        },
        {
            "id": "held_out_tabs_01",
            "openui": (
                'root = Stack([panel], "column")\n'
                'overview = TextContent(":held.tabs.overview")\n'
                'details = TextContent(":held.tabs.details")\n'
                'tab1 = TabItem("$0", ":held.tabs.tab1", [overview])\n'
                'tab2 = TabItem("$1", ":held.tabs.tab2", [details])\n'
                "panel = Tabs([tab1, tab2])"
            ),
            "placeholders": [
                ":held.tabs.overview",
                ":held.tabs.details",
                ":held.tabs.tab1",
                ":held.tabs.tab2",
            ],
        },
        {
            "id": "held_out_settings_01",
            "openui": (
                'root = Stack([notify, volume], "column")\n'
                'notify = SwitchItem(":held.settings.notify", ":held.settings.notify.desc", "$0")\n'
                'volume = Slider("$1", "continuous", 0, 100, 1, [40], ":held.settings.volume")'
            ),
            "placeholders": [
                ":held.settings.notify",
                ":held.settings.notify.desc",
                ":held.settings.volume",
            ],
        },
    ],
    "ood": [
        {
            "id": "ood_dashboard_01",
            "openui": (
                'root = Stack([header, rule, row], "column")\n'
                'header = TextContent(":ood.dash.header")\n'
                "rule = Separator()\n"
                'm1_title = TextContent(":ood.dash.m1.title")\n'
                'm1_body = TextContent(":ood.dash.m1.body")\n'
                "m1 = Card([m1_title, m1_body])\n"
                'm2_title = TextContent(":ood.dash.m2.title")\n'
                'm2_body = TextContent(":ood.dash.m2.body")\n'
                "m2 = Card([m2_title, m2_body])\n"
                'row = Stack([m1, m2], "row")'
            ),
            "placeholders": [
                ":ood.dash.header",
                ":ood.dash.m1.title",
                ":ood.dash.m1.body",
                ":ood.dash.m2.title",
                ":ood.dash.m2.body",
            ],
        },
        {
            "id": "ood_gallery_01",
            "openui": (
                'root = Stack([shot, caption, note, cta], "column")\n'
                'shot = ImageBlock(":ood.gallery.img", ":ood.gallery.alt")\n'
                'caption = TextContent(":ood.gallery.caption")\n'
                'note = Callout("info", ":ood.gallery.hint.title", ":ood.gallery.hint.body")\n'
                'cta = Button(":ood.gallery.cta")'
            ),
            "placeholders": [
                ":ood.gallery.img",
                ":ood.gallery.alt",
                ":ood.gallery.caption",
                ":ood.gallery.hint.title",
                ":ood.gallery.hint.body",
                ":ood.gallery.cta",
            ],
        },
        {
            "id": "ood_modal_01",
            "openui": (
                'root = Stack([dialog], "column")\n'
                'body = TextContent(":ood.modal.body")\n'
                'confirm = Button(":ood.modal.confirm")\n'
                "actions = Buttons([confirm])\n"
                'dialog = Modal(":ood.modal.title", true, [body, actions])'
            ),
            "placeholders": [
                ":ood.modal.body",
                ":ood.modal.confirm",
                ":ood.modal.title",
            ],
        },
        {
            "id": "ood_auth_01",
            "openui": (
                'root = Stack([name, email, create], "column")\n'
                'name = Input("$0", ":ood.auth.name", "text")\n'
                'email = Input("$1", ":ood.auth.email", "email")\n'
                'create = Button(":ood.auth.create")'
            ),
            "placeholders": [":ood.auth.name", ":ood.auth.email", ":ood.auth.create"],
        },
    ],
}


# (a) analyze_reachability's new variant parameter --------------------------


def test_tree_edit_variant_is_the_registered_variant() -> None:
    # TREE_EDIT_VARIANT must be pulled from the SLM-422 registry, not a local
    # literal, so its fingerprint always matches what
    # scripts.verify_decode_invariants certifies.
    assert TREE_EDIT_VARIANT.variant_id == "tree_edit_diffusion"
    assert TREE_EDIT_VARIANT in VARIANTS


@pytest.mark.parametrize(
    "target,slots",
    [
        ('root = Stack([cta], "column")\ncta = Button(":cta.label")', [":cta.label"]),
        (
            'root = Stack([card], "column")\ncard = Card([], "column")',
            [":x"],
        ),
        (SEED, [":x"]),
    ],
)
def test_variant_defaults_to_tree_edit_and_is_bitwise_identical(
    target: str, slots: list[str]
) -> None:
    """The regression guard on the refactor itself: a call that omits
    ``variant`` and a call that passes the default explicitly must return the
    identical case in every field (verdict, reason_code, edit_lower_bound,
    path) -- the two new details keys (variant_id,
    action_alphabet_fingerprint) are attribution metadata, never a behavior
    change, so they are excluded from this comparison and checked separately
    below."""
    implicit = analyze_reachability(SEED, target, slot_inventory=slots)
    explicit = analyze_reachability(
        SEED, target, slot_inventory=slots, variant=TREE_EDIT_VARIANT
    )
    assert implicit.verdict == explicit.verdict
    assert implicit.reason_code == explicit.reason_code
    assert implicit.edit_lower_bound == explicit.edit_lower_bound
    assert implicit.path == explicit.path
    for case in (implicit, explicit):
        assert case.details["variant_id"] == "tree_edit_diffusion"
        assert (
            case.details["action_alphabet_fingerprint"]
            == TREE_EDIT_VARIANT.action_alphabet_fingerprint
        )


def test_non_tree_edit_variant_raises_rather_than_silently_relabeling() -> None:
    other = next(v for v in VARIANTS if v.variant_id != "tree_edit_diffusion")
    with pytest.raises(NotImplementedError):
        analyze_reachability(SEED, SEED, slot_inventory=[":x"], variant=other)


# (b) frozen-corpus regression guard vs the archived SLM-305 report ---------


def test_frozen_smoke_held_out_ood_reproduce_archived_report_exactly() -> None:
    """smoke/held_out/ood are unchanged since iter-slm305-edit-language-
    20260724.md was generated (verified: this fixture's ids/openui/
    placeholders match test_seeds.jsonl verbatim at the time this test was
    written). At the archived report's own parameters (max_edits=8,
    node_budget=120, mode=extended), today's analyzer must reproduce its
    reachable_fraction, decided/unknown counts, and reason histogram for
    exactly these three suites. `adversarial`, `rico`, and `train` are
    excluded: the on-disk corpora for those have already drifted since
    2026-07-24 (rico grew 6->35 records; train's corpus is a gitignored
    generated artifact) for reasons unrelated to this issue's changes --
    the same drift scripts/run_var1_01_set_property_probe.py's own report
    documents for its Arm A."""
    payload = build_report(
        _FROZEN_SMOKE_HELD_OUT_OOD,
        max_edits=8,
        node_budget=120,
        generated_at="2026-07-25T00:00:00Z",
        mode="extended",
    )
    archived = json.loads(_ARCHIVED_DOC.read_text(encoding="utf-8"))
    for suite in ("smoke", "held_out", "ood"):
        live = payload["suites"][suite]
        arch = archived["suites"][suite]
        assert live["n_cases"] == arch["n_cases"], suite
        assert live["n_decided"] == arch["n_decided"], suite
        assert live["n_unknown_budget"] == arch["n_unknown_budget"], suite
        assert live["reachable_fraction"] == arch["reachable_fraction"], suite
        assert live["reason_histogram"] == arch["reason_histogram"], suite
        assert [c["verdict"] for c in live["cases"]] == [
            c["verdict"] for c in arch["cases"]
        ], suite
        assert [c["reason_code"] for c in live["cases"]] == [
            c["reason_code"] for c in arch["cases"]
        ], suite


# (c) ReachabilityMatrixV1 ---------------------------------------------------


def _tiny_matrix() -> dict[str, object]:
    return build_matrix(
        max_edits=8,
        node_budget=120,
        generated_at="2026-07-25T00:00:00Z",
        corpora=_FROZEN_SMOKE_HELD_OUT_OOD,
    )


def test_matrix_rows_are_every_registered_variant() -> None:
    payload = _tiny_matrix()
    assert set(payload["rows"]) == {entry.variant_id for entry in VARIANTS}
    assert set(payload["matrix"].keys()) == set(payload["rows"])


def test_matrix_columns_are_the_six_standard_suites() -> None:
    payload = _tiny_matrix()
    assert payload["columns"] == list(SUITES)
    for row in payload["matrix"].values():
        assert set(row.keys()) == set(SUITES)


def test_tree_edit_row_is_measured_or_corpus_unavailable_never_not_measured() -> None:
    payload = _tiny_matrix()
    row = payload["matrix"]["tree_edit_diffusion"]
    for suite, cell in row.items():
        assert cell["status"] in {STATUS_MEASURED, STATUS_CORPUS_UNAVAILABLE}, suite
        assert cell["reason"] is None
        assert (
            cell["alphabet_fingerprint"] == TREE_EDIT_VARIANT.action_alphabet_fingerprint
        )
    # The frozen fixture only supplies smoke/held_out/ood; the rest have no
    # corpus and must be corpus_unavailable, never zero-reachable.
    for suite in ("train", "adversarial", "rico"):
        assert row[suite]["status"] == STATUS_CORPUS_UNAVAILABLE
        assert row[suite]["reachable_fraction"] is None
    for suite in ("smoke", "held_out", "ood"):
        assert row[suite]["status"] == STATUS_MEASURED
        assert row[suite]["reachable_fraction"] == 0.0


def test_unmeasured_variants_are_not_measured_with_real_distinct_reasons() -> None:
    payload = _tiny_matrix()
    for variant_id in ("repl_operators", "twotower_prompt_ast"):
        row = payload["matrix"][variant_id]
        variant = next(v for v in VARIANTS if v.variant_id == variant_id)
        for suite, cell in row.items():
            assert cell["status"] == STATUS_NOT_MEASURED, suite
            assert cell["reachable_fraction"] is None
            assert cell["decided"] is None
            assert cell["unknown_budget"] is None
            assert cell["reason"] == NOT_MEASURED_REASONS[variant_id]
            assert cell["alphabet_fingerprint"] == variant.action_alphabet_fingerprint
    # The two unmeasured variants' reasons must be genuinely distinct (never
    # a copy-pasted placeholder) and neither may borrow the tree-edit row's
    # fraction or histogram.
    assert (
        NOT_MEASURED_REASONS["repl_operators"]
        != NOT_MEASURED_REASONS["twotower_prompt_ast"]
    )


def test_not_measured_never_inferred_from_tree_edit_row() -> None:
    """Even when the tree-edit row is fully decided/measured for a suite,
    the unmeasured rows for that same suite stay NOT_MEASURED -- nothing
    copies the tree-edit fraction across rows."""
    payload = _tiny_matrix()
    for suite in ("smoke", "held_out", "ood"):
        tree_edit_fraction = payload["matrix"]["tree_edit_diffusion"][suite][
            "reachable_fraction"
        ]
        assert tree_edit_fraction is not None
        for variant_id in ("repl_operators", "twotower_prompt_ast"):
            cell = payload["matrix"][variant_id][suite]
            assert cell["reachable_fraction"] is None
            assert cell["status"] == STATUS_NOT_MEASURED


def test_render_markdown_includes_not_measured_and_measured_cells() -> None:
    payload = _tiny_matrix()
    md = render_markdown(payload)
    assert "NOT_MEASURED" in md
    assert "corpus_unavailable" in md
    assert "`repl_operators`" in md
    assert "`tree_edit_diffusion`" in md
    assert "`twotower_prompt_ast`" in md
    # Every NOT_MEASURED reason string appears verbatim in the rendered doc.
    for reason in NOT_MEASURED_REASONS.values():
        assert reason in md


def test_matrix_is_deterministic() -> None:
    first = _tiny_matrix()
    second = _tiny_matrix()
    first.pop("version_stamp")
    second.pop("version_stamp")
    assert first == second
