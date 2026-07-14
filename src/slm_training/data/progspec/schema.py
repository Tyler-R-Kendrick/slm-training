"""ProgramSpec — the canonical program is the root dataset object.

Every training row is a *derivative* of a validated OpenUI program (a ProgramSpec).
Prompts, corruptions, edits, and renders are lineage-linked to one ProgramSpec and
share its ``split_group_id`` so isomorphic programs (and all their derivatives) land
on the same side of a train/eval split — permutations and paraphrases are never
counted as independent examples, and no derivative leaks across a split boundary.

Derivatives always project down to a plain :class:`ExampleRecord` (task token and
before/patch fold into ``prompt``/``meta``) — there is no new wire type, so the
tokenizer, eval, and ship-gates are untouched.

Re-scope note: targets the installed OpenUI 0.2.x language (see
``dsl.language_contract``); v0.5 constructs (state/query/mutation/action) attach
here as a contract version bump when a package ships.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from slm_training.data.leakage import fingerprint_openui_structure
from slm_training.dsl.language_contract import contract_id as current_contract_id
from slm_training.dsl.placeholders import extract_placeholders
from slm_training.dsl.schema import ExampleRecord

# --- Task tokens (meta["task"]). --------------------------------------------
# GENERATE is the implicit default for plain prompt->program rows; the rest are
# named so one denoiser can specialize. Identity / plain undo-redo / canonical
# formatting are handled deterministically outside the model — only a minority of
# NOOP "restraint" rows are trained.
TASK_GENERATE = "GENERATE"
TASK_REPAIR_MINIMAL = "REPAIR_MINIMAL"
TASK_CANONICALIZE = "CANONICALIZE"
TASK_COMPLETE = "COMPLETE"
TASK_INPAINT = "INPAINT"
TASK_PATCH = "PATCH"
TASK_APPLY_PATCH = "APPLY_PATCH"
TASK_VISUAL_GENERATE = "VISUAL_GENERATE"
TASK_VISUAL_PATCH = "VISUAL_PATCH"
TASK_SEMANTIC_PATCH = "SEMANTIC_PATCH"
TASK_NOOP = "NOOP"

TASKS = frozenset(
    {
        TASK_GENERATE,
        TASK_REPAIR_MINIMAL,
        TASK_CANONICALIZE,
        TASK_COMPLETE,
        TASK_INPAINT,
        TASK_PATCH,
        TASK_APPLY_PATCH,
        TASK_VISUAL_GENERATE,
        TASK_VISUAL_PATCH,
        TASK_SEMANTIC_PATCH,
        TASK_NOOP,
    }
)

# --- Verification tiers (meta["tier"]). -------------------------------------
TIER_GOLD = "gold"
TIER_SILVER = "silver"
TIER_BRONZE = "bronze"
TIER_QUARANTINE = "quarantine"


def structural_family_id(openui: str) -> str:
    """Structural identity: isomorphic layouts (style / placeholder / binder
    normalized) share one id. Reuses the train/eval leakage fingerprint so the
    dataset's family notion and its decontamination notion can never diverge."""
    return fingerprint_openui_structure(openui)


def resolve_split_group_id(openui: str, *, override: str | None = None) -> str:
    """Split-assignment key. Defaults to the structural family so isomorphic
    programs co-locate; ``override`` pins a coarser group (e.g. a source page or
    an edit trajectory) so every one of its variants shares one split."""
    return override or structural_family_id(openui)


@dataclass(frozen=True)
class ProgramSpec:
    """A validated OpenUI program — the root object every derivative hangs off."""

    openui: str
    prompt: str | None = None
    placeholders: tuple[str, ...] = ()
    design_md: str | None = None
    source_family: str = "programspec_generated"
    tier: str = TIER_SILVER
    contract: str = ""
    # Coarser split group (source page / trajectory). Defaults to structural family.
    split_override: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.openui or not self.openui.strip():
            raise ValueError("ProgramSpec.openui must be non-empty")
        if not self.placeholders:
            object.__setattr__(
                self, "placeholders", tuple(extract_placeholders(self.openui))
            )
        if not self.contract:
            object.__setattr__(self, "contract", current_contract_id())

    @property
    def program_family_id(self) -> str:
        return structural_family_id(self.openui)

    @property
    def split_group_id(self) -> str:
        return resolve_split_group_id(self.openui, override=self.split_override)

    @classmethod
    def from_record(cls, record: ExampleRecord) -> ProgramSpec:
        meta = dict(record.meta or {})
        return cls(
            openui=record.openui,
            prompt=record.prompt,
            placeholders=tuple(record.placeholders),
            design_md=record.design_md,
            source_family=str(
                meta.get("source_family") or record.source or "programspec_generated"
            ),
            tier=str(meta.get("tier") or TIER_SILVER),
            contract=str(meta.get("contract_id") or ""),
            split_override=meta.get("split_group_id"),
        )

    def emit_record(
        self,
        *,
        id: str,
        task: str = TASK_GENERATE,
        prompt: str | None = None,
        openui: str | None = None,
        split: str = "train",
        parent_id: str | None = None,
        abstraction_level: str | None = None,
        synth: str | None = None,
        extra_meta: dict[str, Any] | None = None,
    ) -> ExampleRecord:
        """Project a derivative of this program into a canonical ``ExampleRecord``.

        Every derivative inherits the parent's ``split_group_id`` (split-before-
        derive) and ``contract_id``; task / before / patch fold into
        ``prompt``/``meta``. ``openui`` overrides the target (e.g. the *after*
        program of an edit, or the repaired program); it defaults to this program.
        """
        if task not in TASKS:
            raise ValueError(f"unknown task {task!r}; expected one of {sorted(TASKS)}")
        target = openui if openui is not None else self.openui
        meta: dict[str, Any] = {
            "task": task,
            "contract_id": self.contract or current_contract_id(),
            "split_group_id": self.split_group_id,
            "program_family_id": self.program_family_id,
            "tier": self.tier,
            "source_family": self.source_family,
        }
        if parent_id:
            meta["parent_id"] = parent_id
        if synth:
            meta["synth"] = synth
        if abstraction_level:
            meta["abstraction_level"] = abstraction_level
        if self.provenance:
            meta["provenance"] = dict(self.provenance)
        if extra_meta:
            meta.update(extra_meta)
        return ExampleRecord(
            id=id,
            prompt=prompt if prompt is not None else (self.prompt or ""),
            openui=target,
            placeholders=list(extract_placeholders(target)),
            split=split,
            source=self.source_family,
            meta=meta,
            design_md=self.design_md,
        )


def assign_split_groups(records: list[ExampleRecord]) -> list[ExampleRecord]:
    """Stamp ``meta['split_group_id']`` so every record shares its root parent's
    group (split-before-derive). Structure-changing augments therefore stay on the
    same split as the program they derive from, not just verbatim paraphrases.

    Idempotent and deterministic; leaves an explicit override in place.
    """
    by_id = {record.id: record for record in records}
    for record in records:
        meta = dict(record.meta or {})
        if meta.get("split_group_id"):
            continue
        root_id = str(meta.get("root_parent_id") or record.id)
        root = by_id.get(root_id, record)
        meta["split_group_id"] = structural_family_id(root.openui)
        record.meta = meta
    return records
