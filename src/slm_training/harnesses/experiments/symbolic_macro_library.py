"""RSP-005 (SLM-480): prospective certified macro/library-learning
experiment (EXP-SR-9).

Mines recurring solved symbolic-regression subtrees into a small library of
closed (0-arity), content-addressed macros, using an MDL-style
compression objective transplanted from ``data.macro_induction._net_gain``
(token spans -> AST subtree size), then measures whether that library
reduces structural node count *prospectively*, on a frozen held-out slice
the library was never mined from -- retrospective corpus compression alone
is explicitly not success (SLM-480's own falsifier).

Design choices, and how each answers SLM-480's acceptance criteria:

* **No new IR node, no new legality path.** A macro is a lookup-table entry
  ``macro_id -> canonical SymbolicExprNode`` (stored as its
  ``symbolic_expr_ir.serialize_expr`` string). "Expansion" is never spliced
  into the live IR as a call-node; it only changes how *evidence* is
  counted (:func:`instrumented_node_count`). Nothing in
  ``dsl/symbolic_regression_pack.py`` or any production canonicalization
  path imports this module, matching RSP-004's own default-off precedent.
* **Closed macros only (no argument binding).** Every admitted macro is a
  literal, self-contained recurring canonical subtree -- not a parametrized
  template. This is a deliberate scope-limitation versus a "true"
  anti-unified macro system: it sidesteps arity/type-signature ambiguity by
  construction (arity is always 0), at the cost of not compressing
  structurally-similar-but-not-identical subtrees. claim_class stays
  ``"fixture"``; this is not a promotion candidate.
* **Expansion is deterministic and legality-preserving.**
  :func:`expand_macro` is exactly ``symbolic_expr_ir.parse_expr`` of the
  stored canonical string; :func:`verify_expansion_equivalence` finds a real
  occurrence of each admitted macro, splices the reconstruction back in, and
  independently re-verifies numerically (NaN-aware, mirroring RSP-004's
  ``verify_rewrite_numerically``) rather than trusting substitution by
  construction alone.
* **Token/ID versioning never moves historical identities silently.**
  ``MacroEntryV1.macro_id`` is ``sha256(expansion_expr)``-derived
  (:func:`_macro_id_for`): content-addressed, so admitting or retiring a
  macro never renumbers another macro's id -- there is no sequential
  counter to shift.
* **No held-out/eval AST influences the learned library.**
  :func:`mine_macro_candidates` only ever accepts ``corpus_split=="train"``
  records; :func:`run_campaign` evaluates the mined libraries exclusively
  against ``corpus_split=="test"`` records, matching exp-sr-9's own
  ``leakage_control``.
* **Result remains experimental until prospective evidence passes.**
  :class:`Rsp005CampaignV1` fixes ``claim_class="fixture"`` in
  ``__post_init__``; ``run_campaign`` never sets ``promotion: True``.

Documented divergence (ownership_map.json): the pre-registered
``downstream_extension_map`` row for RSP-005/SLM-480 predicted
``new_owner_justified: false`` against ``dsl_pack_registry_canonical`` /
``certified_completion_artifact`` / ``experiment_campaign`` -- i.e. no new
owner beyond those three. The macro-library contract itself
(``MacroLibraryV1``/``MacroEntryV1``/mining and selection functions) is not
literally defined in ``dsl/pack.py`` or
``dsl/grammar/fastpath/completion_artifact.py``, so per this map's own
sibling-file rule it is registered as its own ``symbolic_macro_library``
subsystem instead -- same precedent as SGS-008/SRP-004/005/007/008/009/
RSP-004. ``experiment_campaign`` *is* genuinely reused directly (no fork):
:class:`Rsp005CampaignV1` builds a real
``autoresearch.experiment_campaign.ExperimentCampaignV1`` and runs it
through the shared ``autoresearch.storage.CampaignStore``, exactly like
SIE-002/RSP-004. This module also does not touch, extend, or compete with
``data/macro_induction.py`` / ``models/dsl_tokenizer.py`` -- that is a
separate, production, compiler-hard, version-governed macro system over the
*OpenUI DSL token stream*; this module mines macros over the *unrelated*
symbolic-regression AST domain and is evaluation-only.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from slm_training.autoresearch.experiment_campaign import (
    ArtifactRequirementV1,
    CampaignArmV1,
    CampaignControlV1,
    CampaignEndpointV1,
    CampaignGateV1,
    ExperimentCampaignV1,
    MultiplicityFamilyV1,
    campaign_manifest_sha256,
)
from slm_training.autoresearch.schemas import CampaignBudget, CampaignSpec
from slm_training.autoresearch.storage import CampaignStore
from slm_training.dsl.symbolic_expr_canonicalize import canonicalize_expr
from slm_training.dsl.symbolic_expr_corpus import (
    CORPUS_SCHEMA_VERSION,
    CorpusGenerationConfigV1,
    CorpusRecordV1,
    audit_corpus_leakage,
    generate_corpus,
)
from slm_training.dsl.symbolic_expr_evaluator import evaluate_vectorized
from slm_training.dsl.symbolic_expr_ir import (
    OpApply,
    SymbolicExprNode,
    expr_depth_and_node_count,
    parse_expr,
    parse_problem_expr,
    serialize_expr,
)
from slm_training.harness_core.lineage.records import content_sha
from slm_training.harnesses.experiments.exp_sr_catalogue import (
    exp_sr_campaign,
    plan_only_manifest,
)
from slm_training.harnesses.train_data.split_policy import RootFamilySplitPolicyV1
from slm_training.levers import MAX_RUN_MINUTES

CATALOGUE_ID = "exp-sr-9"
PRIMARY_METRIC = "macro_library_size_reduction_rate"
MACRO_LIBRARY_SCHEMA_VERSION = "v1"
ARM_IDS = ("no_macros", "frequency_macros", "mdl_macros")
_SELECTION_POLICIES = frozenset({"none", "frequency_only", "mdl_net_gain"})


def _macro_id_for(canonical_key: str) -> str:
    """Content-addressed macro identity: never depends on library
    membership, enumeration order, or admission/retirement of any other
    macro (SLM-480: 'Token/ID versioning does not move historical
    identities silently')."""
    return "macro_" + hashlib.sha256(canonical_key.encode("utf-8")).hexdigest()[:16]


def _net_gain(node_count: int, frequency: int) -> int:
    """MDL-style compression objective, transplanted from
    ``data.macro_induction._net_gain`` (token spans) onto AST subtree size:
    replacing every occurrence with one call-site node costs ``frequency``
    call-site units plus a one-time ``node_count`` definition cost, against
    a ``frequency * node_count`` uncompressed baseline."""
    return frequency * (node_count - 1) - node_count


@dataclass(frozen=True)
class MacroCandidateV1:
    """One recurring canonical subtree observed while mining, before any
    admission decision."""

    canonical_key: str
    node_count: int
    frequency: int
    net_gain_nodes: int

    def __post_init__(self) -> None:
        if self.node_count < 2:
            raise ValueError("a macro candidate must span at least 2 IR nodes")
        if self.frequency < 1:
            raise ValueError("frequency must be positive")
        if _net_gain(self.node_count, self.frequency) != self.net_gain_nodes:
            raise ValueError(
                "net_gain_nodes inconsistent with frequency*(node_count-1)-node_count"
            )


@dataclass(frozen=True)
class MacroEntryV1:
    """One admitted, closed (0-arity) macro: a content-addressed id bound
    to its own literal canonical expansion. No argument binding exists by
    construction, so there is no arity/type-signature ambiguity to
    validate beyond internal self-consistency."""

    macro_id: str
    expansion_expr: str
    node_count: int
    frequency: int
    net_gain_nodes: int

    def __post_init__(self) -> None:
        if self.node_count < 2:
            raise ValueError("a macro must span at least 2 IR nodes")
        if self.frequency < 1:
            raise ValueError("frequency must be positive")
        if _net_gain(self.node_count, self.frequency) != self.net_gain_nodes:
            raise ValueError(
                "net_gain_nodes inconsistent with frequency*(node_count-1)-node_count"
            )
        if self.macro_id != _macro_id_for(self.expansion_expr):
            raise ValueError(
                "macro_id must be sha256(expansion_expr)-derived (content-addressed)"
            )


def _macro_library_digest(macros: tuple[MacroEntryV1, ...]) -> str:
    return content_sha([asdict(entry) for entry in macros])


@dataclass(frozen=True)
class MacroLibraryV1:
    """A fully admitted macro library: sorted, deduplicated, and
    tamper-evident (``library_digest`` is recomputed and checked against
    ``macros`` on every construction)."""

    schema_version: str
    selection_policy: str
    macros: tuple[MacroEntryV1, ...]
    library_digest: str

    def __post_init__(self) -> None:
        if self.schema_version != MACRO_LIBRARY_SCHEMA_VERSION:
            raise ValueError(f"unsupported macro library schema: {self.schema_version!r}")
        if self.selection_policy not in _SELECTION_POLICIES:
            raise ValueError(f"unsupported selection_policy: {self.selection_policy!r}")
        ids = [entry.macro_id for entry in self.macros]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate macro_id in library")
        if tuple(sorted(self.macros, key=lambda entry: entry.macro_id)) != self.macros:
            raise ValueError("macros must be sorted by macro_id for determinism")
        if _macro_library_digest(self.macros) != self.library_digest:
            raise ValueError("library_digest does not match macros (tamper or construction bug)")


def build_macro_library(
    macros: Sequence[MacroEntryV1], *, selection_policy: str
) -> MacroLibraryV1:
    ordered = tuple(sorted(macros, key=lambda entry: entry.macro_id))
    return MacroLibraryV1(
        schema_version=MACRO_LIBRARY_SCHEMA_VERSION,
        selection_policy=selection_policy,
        macros=ordered,
        library_digest=_macro_library_digest(ordered),
    )


def empty_macro_library() -> MacroLibraryV1:
    """The ``no_macros`` control arm's library -- must always yield exactly
    zero size reduction (the campaign's ``no_op_control``)."""
    return build_macro_library((), selection_policy="none")


def _iter_subtrees(node: SymbolicExprNode):
    yield node
    if isinstance(node, OpApply):
        for arg in node.args:
            yield from _iter_subtrees(arg)


def mine_macro_candidates(
    train_records: Sequence[CorpusRecordV1], *, min_node_count: int = 2
) -> tuple[MacroCandidateV1, ...]:
    """Enumerate recurring canonical subtrees over train-split solved ASTs
    only. Callers must never pass validation/test records here -- this is
    the module's entire leakage-control surface (SLM-480 acceptance
    criterion: 'No held-out/eval AST influences learned library.')."""
    counts: dict[str, int] = {}
    node_counts: dict[str, int] = {}
    for record in train_records:
        tree = parse_problem_expr(record.problem, record.oracle.canonical_expr)
        for subtree in _iter_subtrees(tree):
            if not isinstance(subtree, OpApply):
                continue
            canonical = canonicalize_expr(subtree)
            key = serialize_expr(canonical)
            _, node_count = expr_depth_and_node_count(canonical)
            if node_count < min_node_count:
                continue
            counts[key] = counts.get(key, 0) + 1
            node_counts[key] = node_count
    return tuple(
        MacroCandidateV1(
            canonical_key=key,
            node_count=node_counts[key],
            frequency=frequency,
            net_gain_nodes=_net_gain(node_counts[key], frequency),
        )
        for key, frequency in counts.items()
    )


def select_macros_by_mdl(
    candidates: Sequence[MacroCandidateV1],
    *,
    min_frequency: int = 2,
    min_gain_nodes: int = 1,
    max_macros: int = 16,
) -> MacroLibraryV1:
    """The treatment arm: admit by the MDL net-gain objective."""
    eligible = [
        c for c in candidates if c.frequency >= min_frequency and c.net_gain_nodes >= min_gain_nodes
    ]
    eligible.sort(key=lambda c: (-c.net_gain_nodes, -c.node_count, c.canonical_key))
    chosen = eligible[:max_macros]
    macros = tuple(
        MacroEntryV1(
            macro_id=_macro_id_for(c.canonical_key),
            expansion_expr=c.canonical_key,
            node_count=c.node_count,
            frequency=c.frequency,
            net_gain_nodes=c.net_gain_nodes,
        )
        for c in chosen
    )
    return build_macro_library(macros, selection_policy="mdl_net_gain")


def select_macros_by_frequency(
    candidates: Sequence[MacroCandidateV1], *, min_frequency: int = 2, max_macros: int = 16
) -> MacroLibraryV1:
    """The weaker control arm: admit by raw frequency alone, ignoring the
    compression-aware net-gain objective."""
    eligible = [c for c in candidates if c.frequency >= min_frequency]
    eligible.sort(key=lambda c: (-c.frequency, -c.node_count, c.canonical_key))
    chosen = eligible[:max_macros]
    macros = tuple(
        MacroEntryV1(
            macro_id=_macro_id_for(c.canonical_key),
            expansion_expr=c.canonical_key,
            node_count=c.node_count,
            frequency=c.frequency,
            net_gain_nodes=c.net_gain_nodes,
        )
        for c in chosen
    )
    return build_macro_library(macros, selection_policy="frequency_only")


def _macro_lookup(library: MacroLibraryV1) -> dict[str, MacroEntryV1]:
    return {entry.expansion_expr: entry for entry in library.macros}


def instrumented_node_count(node: SymbolicExprNode, library: MacroLibraryV1) -> int:
    """Node count after greedy top-down macro substitution: a matched
    subtree collapses to a single call-site node and is never recursed
    into, so an admitted macro's own descendant macros never double-count
    savings -- this makes mining order-independent: candidates never need
    to reason about nesting, only this evaluation step does."""
    lookup = _macro_lookup(library)

    def _walk(current: SymbolicExprNode) -> int:
        key = serialize_expr(canonicalize_expr(current))
        if key in lookup:
            return 1
        if isinstance(current, OpApply):
            return 1 + sum(_walk(arg) for arg in current.args)
        return 1

    return _walk(node)


def macro_library_size_reduction_rate(
    trees: Sequence[SymbolicExprNode], library: MacroLibraryV1
) -> float:
    """The campaign's primary metric: fractional node-count reduction over
    ``trees`` under ``library``. Callers are responsible for passing only
    held-out trees when a prospective (non-retrospective) claim is
    intended."""
    baseline_total = 0
    instrumented_total = 0
    for tree in trees:
        _, node_count = expr_depth_and_node_count(tree)
        baseline_total += node_count
        instrumented_total += instrumented_node_count(tree, library)
    if baseline_total == 0:
        return 0.0
    return (baseline_total - instrumented_total) / baseline_total


def expand_macro(entry: MacroEntryV1, *, num_variables: int) -> SymbolicExprNode:
    """Deterministically reconstruct a macro's canonical expansion. Every
    macro is closed (0-arity): no argument binding, no free variable beyond
    what the stored expansion already names, so reconstruction is exactly
    ``parse_expr`` of the stored string -- nothing more."""
    return parse_expr(entry.expansion_expr, num_variables=num_variables)


def _find_occurrence(tree: SymbolicExprNode, canonical_key: str) -> SymbolicExprNode | None:
    if serialize_expr(canonicalize_expr(tree)) == canonical_key:
        return tree
    if isinstance(tree, OpApply):
        for arg in tree.args:
            found = _find_occurrence(arg, canonical_key)
            if found is not None:
                return found
    return None


def _replace_first_occurrence(
    tree: SymbolicExprNode, canonical_key: str, replacement: SymbolicExprNode
) -> SymbolicExprNode:
    if serialize_expr(canonicalize_expr(tree)) == canonical_key:
        return replacement
    if isinstance(tree, OpApply):
        new_args = []
        replaced = False
        for arg in tree.args:
            if not replaced:
                new_arg = _replace_first_occurrence(arg, canonical_key, replacement)
                if new_arg is not arg:
                    replaced = True
                new_args.append(new_arg)
            else:
                new_args.append(arg)
        return OpApply(operator=tree.operator, args=tuple(new_args))
    return tree


def verify_expansion_equivalence(
    library: MacroLibraryV1,
    records: Sequence[CorpusRecordV1],
    *,
    rng_seed: int,
    n_samples: int = 8,
    sample_domain: tuple[float, float] = (-2.0, 2.0),
) -> tuple[dict[str, Any], ...]:
    """For every admitted macro, find one real occurrence among ``records``,
    expand it back from its stored string, splice the reconstruction into
    the original tree, and independently re-verify numerically (NaN-aware)
    that nothing moved -- SLM-480's 'expansion is deterministic and
    legality-preserving' acceptance criterion, mirroring RSP-004's
    ``verify_rewrite_numerically`` pattern rather than trusting
    substitution by construction alone."""
    rng = random.Random(rng_seed)
    reports: list[dict[str, Any]] = []
    for entry in library.macros:
        occurrence_tree: SymbolicExprNode | None = None
        source_record: CorpusRecordV1 | None = None
        for record in records:
            tree = parse_problem_expr(record.problem, record.oracle.canonical_expr)
            if _find_occurrence(tree, entry.expansion_expr) is not None:
                occurrence_tree = tree
                source_record = record
                break
        if occurrence_tree is None or source_record is None:
            reports.append(
                {"macro_id": entry.macro_id, "verified": False, "reason": "no occurrence found"}
            )
            continue
        num_variables = len(source_record.problem.variables)
        expansion = expand_macro(entry, num_variables=num_variables)
        reconstructed = _replace_first_occurrence(occurrence_tree, entry.expansion_expr, expansion)
        variables = np.asarray(
            [[rng.uniform(*sample_domain) for _ in range(num_variables)] for _ in range(n_samples)]
        )
        coefficients = (
            np.asarray(source_record.oracle.true_coefficients)
            if source_record.oracle.true_coefficients
            else None
        )
        original_eval = evaluate_vectorized(
            occurrence_tree,
            variables=variables,
            coefficients=coefficients,
            domain_policy="allow_nonfinite",
        )
        reconstructed_eval = evaluate_vectorized(
            reconstructed,
            variables=variables,
            coefficients=coefficients,
            domain_policy="allow_nonfinite",
        )
        equal = bool(
            np.array_equal(
                np.asarray(original_eval.predictions),
                np.asarray(reconstructed_eval.predictions),
                equal_nan=True,
            )
        )
        reports.append(
            {
                "macro_id": entry.macro_id,
                "verified": equal,
                "reason": "ok" if equal else "numeric mismatch after expand/reconstruct round-trip",
                "record_family_id": source_record.family_id,
            }
        )
    return tuple(reports)


# ---------------------------------------------------------------------------
# Campaign wiring (exp-sr-9, claim_class=fixture) -- mirrors SIE-002/RSP-004
# ---------------------------------------------------------------------------


def _default_corpus_config(seed: int) -> CorpusGenerationConfigV1:
    """A small, deterministic in-process corpus with enough repeated
    substructure (limited variable/operator vocabulary, shallow depth) for
    macro mining to have real candidates to find at fixture scale."""
    return CorpusGenerationConfigV1(
        schema_version=CORPUS_SCHEMA_VERSION,
        seed=seed,
        num_problems=80,
        num_variables_range=(1, 2),
        allowed_operators=frozenset({"+", "*", "neg", "sin"}),
        max_basis_depth=3,
        max_coefficient_terms=2,
        coefficient_probability=0.5,
        numeric_domain_policy="reject_nonfinite",
        complexity_budget=24,
        train_rows=4,
        validation_rows=2,
        extrapolation_rows=2,
        train_domain=(-2.0, 2.0),
        extrapolation_domain=(3.0, 5.0),
        coefficient_range=(-2.0, 2.0),
        split_policy=RootFamilySplitPolicyV1(),
    )


@dataclass(frozen=True)
class Rsp005CampaignV1:
    """Frozen campaign parameters. ``claim_class`` is fixed to
    ``"fixture"`` in ``__post_init__``: this experiment never promotes a
    checkpoint (SLM-480: 'Result remains experimental until prospective
    evidence passes.')."""

    campaign_id: str = "rsp005-exp-sr-9-macro-library-fixture"
    corpus_seed: int = 0
    min_frequency: int = 2
    min_gain_nodes: int = 1
    max_macros: int = 16
    seeds: tuple[int, ...] = (0,)
    source_commit: str = "0" * 40
    claim_class: str = "fixture"
    max_wall_minutes: float = float(MAX_RUN_MINUTES)

    def __post_init__(self) -> None:
        if self.min_frequency < 1:
            raise ValueError("min_frequency must be positive")
        if self.min_gain_nodes < 1:
            raise ValueError("min_gain_nodes must be positive")
        if self.max_macros < 1:
            raise ValueError("max_macros must be positive")
        if self.claim_class != "fixture":
            raise ValueError("RSP-005 durable evidence is fixture-only (no promotion)")
        if self.max_wall_minutes <= 0 or self.max_wall_minutes > MAX_RUN_MINUTES:
            raise ValueError("max_wall_minutes must obey MAX_RUN_MINUTES")
        if not self.seeds:
            raise ValueError("seeds must be non-empty")

    @property
    def fingerprint(self) -> str:
        return content_sha(asdict(self))

    def catalogue_manifest(self) -> ExperimentCampaignV1:
        return exp_sr_campaign(CATALOGUE_ID)

    def minimum_effect(self) -> float:
        return float(self.catalogue_manifest().endpoints[0].minimum_effect)

    def manifest(self) -> ExperimentCampaignV1:
        fingerprint = self.fingerprint
        catalogue = self.catalogue_manifest()
        arms = tuple(
            CampaignArmV1(
                arm_id=arm,
                role="control" if arm == "no_macros" else "candidate",
                config_sha256=content_sha({"campaign": fingerprint, "arm": arm}),
            )
            for arm in ARM_IDS
        )
        return ExperimentCampaignV1(
            campaign_id=self.campaign_id,
            experiment_id=CATALOGUE_ID,
            hypothesis=catalogue.hypothesis,
            decision=(
                "Adopt learned macros for later structural-search work only when "
                "the MDL-selected library beats both the no-macro baseline and a "
                "frequency-only control on prospective, train-only-mined, "
                "held-out evidence with zero expansion-equivalence failures; "
                "empty adoption is an honest falsification. No checkpoint or "
                "default promotion from this diagnostic."
            ),
            endpoints=(
                CampaignEndpointV1(
                    endpoint_id=f"{CATALOGUE_ID}_primary",
                    metric=PRIMARY_METRIC,
                    role="primary",
                    direction="increase",
                    minimum_effect=self.minimum_effect(),
                ),
            ),
            arms=arms,
            seeds=self.seeds,
            budget=CampaignBudget(
                max_experiments=len(ARM_IDS) * len(self.seeds),
                max_wall_minutes=self.max_wall_minutes,
            ),
            stopping_rules=(
                "A macro with no real occurrence in the supplied records is never "
                "counted as expansion-verified.",
                "Every record is generated in-process; no subprocess timeout applies.",
                "No GPU / model forward; fixture proxy only.",
            ),
            controls=(
                CampaignControlV1(
                    control_id="no_op_control",
                    description=(
                        "An empty macro library (selection_policy=none) must "
                        "yield exactly zero size reduction."
                    ),
                    kind="quality",
                ),
                CampaignControlV1(
                    control_id="leakage_control",
                    description=(
                        "Macro library is mined exclusively from "
                        "corpus_split=='train' records; corpus_split=='test' "
                        "records are only ever evaluated against, never mined "
                        "from."
                    ),
                    kind="negative",
                ),
            ),
            negative_controls=("leakage_control",),
            multiplicity_families=(
                MultiplicityFamilyV1(
                    family_id=f"{CATALOGUE_ID}_family",
                    hypothesis_ids=(f"{CATALOGUE_ID}_primary",),
                    alpha=0.05,
                ),
            ),
            promotion_gates=(
                CampaignGateV1(
                    gate_id="fixture_only",
                    endpoint_id=f"{CATALOGUE_ID}_primary",
                    operator="ge",
                    threshold=0.0,
                ),
            ),
            rollback_gates=(
                CampaignGateV1(
                    gate_id=f"{CATALOGUE_ID}_kill",
                    endpoint_id=f"{CATALOGUE_ID}_primary",
                    operator="le",
                    threshold=0.0,
                ),
            ),
            artifact_requirements=(
                ArtifactRequirementV1(kind="version_stamp"),
                ArtifactRequirementV1(kind="endpoint_result"),
                ArtifactRequirementV1(kind="paired_examples"),
            ),
            claim_class=self.claim_class,
            source_commit=self.source_commit,
            source_dirty=False,
            author="slm-training",
            created_at="1970-01-01T00:00:00Z",
        )


def run_campaign(campaign: Rsp005CampaignV1, *, root: Path) -> dict[str, Any]:
    """Mine a train-only macro library, evaluate prospectively on the frozen
    test slice, and persist honest evidence under ``CampaignStore``."""
    store = CampaignStore(campaign.campaign_id, root)
    store.initialize(
        CampaignSpec(
            campaign_id=campaign.campaign_id,
            objective="RSP-005 / EXP-SR-9 fixture-scale certified macro/library-learning campaign",
            primary_metric=PRIMARY_METRIC,
            budget=CampaignBudget(
                max_experiments=len(ARM_IDS) * len(campaign.seeds),
                max_wall_minutes=campaign.max_wall_minutes,
            ),
            created_at="1970-01-01T00:00:00Z",
        )
    )
    lock = store.lock_experiment_campaign(campaign.manifest())
    catalogue = campaign.catalogue_manifest()

    config = _default_corpus_config(campaign.corpus_seed)
    corpus = generate_corpus(config)
    leakage = audit_corpus_leakage(corpus)

    train_records = [r for r in corpus.records if r.corpus_split == "train"]
    test_records = [r for r in corpus.records if r.corpus_split == "test"]
    if not train_records or not test_records:
        raise ValueError(
            "corpus produced an empty train or test split -- cannot honestly "
            "measure a prospective, leakage-controlled reduction rate"
        )

    candidates = mine_macro_candidates(train_records, min_node_count=2)
    libraries = {
        "no_macros": empty_macro_library(),
        "frequency_macros": select_macros_by_frequency(
            candidates, min_frequency=campaign.min_frequency, max_macros=campaign.max_macros
        ),
        "mdl_macros": select_macros_by_mdl(
            candidates,
            min_frequency=campaign.min_frequency,
            min_gain_nodes=campaign.min_gain_nodes,
            max_macros=campaign.max_macros,
        ),
    }

    held_out_trees = [
        parse_problem_expr(record.problem, record.oracle.canonical_expr) for record in test_records
    ]
    arm_results = {
        arm_id: {
            "reduction_rate": macro_library_size_reduction_rate(held_out_trees, library),
            "library_size": len(library.macros),
            "library_digest": library.library_digest,
            "selection_policy": library.selection_policy,
        }
        for arm_id, library in libraries.items()
    }

    equivalence = verify_expansion_equivalence(
        libraries["mdl_macros"], corpus.records, rng_seed=campaign.corpus_seed
    )
    equivalence_failures = tuple(report for report in equivalence if not report["verified"])

    no_op_holds = arm_results["no_macros"]["reduction_rate"] == 0.0
    mdl_beats_baseline = (
        arm_results["mdl_macros"]["reduction_rate"] - arm_results["no_macros"]["reduction_rate"]
        >= campaign.minimum_effect()
    )
    mdl_beats_frequency = (
        arm_results["mdl_macros"]["reduction_rate"] >= arm_results["frequency_macros"]["reduction_rate"]
    )
    falsifier_holds = bool(equivalence_failures) or not no_op_holds
    authorized = (
        not falsifier_holds and mdl_beats_baseline and mdl_beats_frequency and leakage.is_clean
    )

    paired_examples = [
        {
            "family_id": test_records[i].family_id,
            "canonical_expr": test_records[i].oracle.canonical_expr,
            "baseline_node_count": expr_depth_and_node_count(held_out_trees[i])[1],
            "mdl_instrumented_node_count": instrumented_node_count(
                held_out_trees[i], libraries["mdl_macros"]
            ),
        }
        for i in range(min(5, len(held_out_trees)))
    ]

    result: dict[str, Any] = {
        "schema": "rsp005_macro_library_fixture_result/v1",
        "campaign_id": campaign.campaign_id,
        "catalogue_id": CATALOGUE_ID,
        "catalogue_manifest_sha256": campaign_manifest_sha256(catalogue),
        "manifest_sha256": lock.manifest_sha256,
        "claim_class": campaign.claim_class,
        "promotion": False,
        "corpus_seed": campaign.corpus_seed,
        "leakage_audit": leakage.to_dict(),
        "n_train_records": len(train_records),
        "n_test_records": len(test_records),
        "candidate_count": len(candidates),
        "arms": arm_results,
        "primary_metric": PRIMARY_METRIC,
        "minimum_effect": campaign.minimum_effect(),
        "expansion_equivalence": list(equivalence),
        "expansion_equivalence_all_verified": not equivalence_failures,
        "falsifier_holds": falsifier_holds,
        "authorized": authorized,
        "scope_disclaimer": (
            "Fixture-scale certified macro/library-learning campaign over a "
            "small deterministic in-process symbolic-regression corpus "
            "(symbolic_expr_corpus.generate_corpus). Macros are closed "
            "(0-arity) recurring canonical subtrees mined exclusively from "
            "corpus_split=='train' records and evaluated on the frozen "
            "corpus_split=='test' slice, never the reverse. claim_class=fixture; "
            "'authorized' is a diagnostic signal for later structural-search "
            "work and never promotes a checkpoint."
        ),
    }
    artifact = store.write_artifact("rsp005_macro_library_result", result)
    store.write_artifact(
        "paired_examples", {"campaign_id": campaign.campaign_id, "examples": paired_examples}
    )
    store.append_event(
        "rsp005_macro_library_completed",
        experiment_id=campaign.campaign_id,
        status="fixture",
        artifact_sha256=artifact.stem,
        detail={
            "manifest_sha256": lock.manifest_sha256,
            "authorized": authorized,
            "mdl_reduction_rate": arm_results["mdl_macros"]["reduction_rate"],
        },
    )
    return result


def plan_only_preview() -> dict[str, Any]:
    return plan_only_manifest(CATALOGUE_ID)


__all__ = [
    "ARM_IDS",
    "CATALOGUE_ID",
    "MACRO_LIBRARY_SCHEMA_VERSION",
    "PRIMARY_METRIC",
    "MacroCandidateV1",
    "MacroEntryV1",
    "MacroLibraryV1",
    "Rsp005CampaignV1",
    "build_macro_library",
    "empty_macro_library",
    "expand_macro",
    "instrumented_node_count",
    "macro_library_size_reduction_rate",
    "mine_macro_candidates",
    "plan_only_preview",
    "run_campaign",
    "select_macros_by_frequency",
    "select_macros_by_mdl",
    "verify_expansion_equivalence",
]
