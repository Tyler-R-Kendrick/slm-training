"""Deterministic generalization slices with existing leakage fingerprints."""

from __future__ import annotations

import itertools
import re
from collections.abc import Iterable, Mapping
from typing import Any

from slm_training.data.leakage import (
    find_leakage,
    fingerprint_openui,
    fingerprint_openui_structure,
    fingerprint_pair,
    fingerprint_prompt,
)
from slm_training.dsl.parser import ParseError, validate
from slm_training.dsl.schema import ExampleRecord
from slm_training.models.tokenizer import tokenize_text

_COMPONENT_RE = re.compile(r"\b([A-Z][A-Za-z0-9]*)\s*\(")
_BINDER_ASSIGN_RE = re.compile(r"(?m)^\s*([a-z_][A-Za-z0-9_]*)\s*=")
_LOWER_NAME_RE = re.compile(r"\b[a-z_][A-Za-z0-9_]*\b")


def _component_combinations(source: str, size: int) -> set[tuple[str, ...]]:
    names = sorted(set(_COMPONENT_RE.findall(source)))
    return set(itertools.combinations(names, size))


def rename_binders(source: str) -> str:
    """Apply a deterministic, string-safe binder permutation (E65 stress)."""
    names = sorted(set(_BINDER_ASSIGN_RE.findall(source)) - {"root"})
    mapping = {name: f"sym_{index}" for index, name in enumerate(names)}
    if not mapping:
        return source
    pieces = re.split(r'("(?:\\.|[^"\\])*")', source)
    for index in range(0, len(pieces), 2):
        pieces[index] = _LOWER_NAME_RE.sub(
            lambda match: mapping.get(match.group(0), match.group(0)), pieces[index]
        )
    return "".join(pieces)


def reorder_declarations(source: str) -> str:
    """Reverse declaration order while preserving every declaration byte."""
    lines = [line for line in source.splitlines() if line.strip()]
    suffix = "\n" if source.endswith("\n") else ""
    return "\n".join(reversed(lines)) + suffix


def schema_transfer_report(
    train_records: Iterable[ExampleRecord],
    held_records: Iterable[ExampleRecord],
    *,
    alien_programs: Mapping[str, Iterable[str]] | None = None,
) -> dict[str, Any]:
    """Leave-family-out and representation-perturbation audit (E65)."""
    train = list(train_records)
    held = list(held_records)
    known = set().union(*(_component_combinations(row.openui, 1) for row in train))
    known_names = {item[0] for item in known}
    held_names = set().union(*(_component_combinations(row.openui, 1) for row in held))
    held_name_set = {item[0] for item in held_names}
    unseen = sorted(held_name_set - known_names)
    leave_one_out = {
        family: sum(
            family in {item[0] for item in _component_combinations(row.openui, 1)}
            for row in held
        )
        for family in unseen
    }

    perturbations: list[dict[str, Any]] = []
    for record in sorted(held, key=lambda row: row.id):
        renamed = rename_binders(record.openui)
        reordered = reorder_declarations(record.openui)
        row = {"id": record.id, "rename_valid": False, "reorder_valid": False}
        try:
            row["rename_valid"] = validate(renamed).root is not None
        except (ParseError, ValueError, RuntimeError):
            pass
        try:
            row["reorder_valid"] = validate(reordered).root is not None
        except (ParseError, ValueError, RuntimeError):
            pass
        perturbations.append(row)

    alien: dict[str, dict[str, Any]] = {}
    if alien_programs:
        from slm_training.dsl.grammar.backends import get_backend

        for dsl, programs in sorted(alien_programs.items()):
            sources = list(programs)
            passed = 0
            errors: list[str] = []
            backend = get_backend(dsl)
            for source in sources:
                try:
                    backend.validate(source)
                    passed += 1
                except Exception as exc:  # noqa: BLE001 - transfer audit records failure
                    errors.append(str(exc).splitlines()[0][:200])
            alien[dsl] = {"n": len(sources), "valid": passed, "errors": errors[:10]}

    return {
        "known_schema_families": sorted(known_names),
        "held_schema_families": sorted(held_name_set),
        "unseen_schema_families": unseen,
        "leave_one_schema_family_out_n": leave_one_out,
        "symbol_rename_valid_rate": (
            sum(bool(row["rename_valid"]) for row in perturbations) / len(perturbations)
            if perturbations
            else None
        ),
        "declaration_reorder_valid_rate": (
            sum(bool(row["reorder_valid"]) for row in perturbations)
            / len(perturbations)
            if perturbations
            else None
        ),
        "perturbations": perturbations,
        "alien_grammar_transfer": alien,
    }


def _tree_depth(value: Any) -> int:
    if isinstance(value, Mapping):
        children = [_tree_depth(child) for child in value.values()]
        return 1 + max(children, default=0)
    if isinstance(value, list):
        return max((_tree_depth(child) for child in value), default=0)
    return 0


def _program_depth(source: str) -> int:
    try:
        return _tree_depth(validate(source).root)
    except (ParseError, ValueError):
        return 0


def _edit_composition(record: ExampleRecord) -> tuple[str, ...] | None:
    edit = (record.meta or {}).get("edit")
    if not isinstance(edit, Mapping):
        return None
    raw = edit.get("operators") or edit.get("operator")
    if isinstance(raw, str):
        return (raw,)
    if isinstance(raw, list):
        return tuple(str(value) for value in raw)
    return None


def _domain(record: ExampleRecord) -> str | None:
    meta = record.meta or {}
    provenance = meta.get("provenance")
    for source in (meta, provenance if isinstance(provenance, Mapping) else {}):
        value = source.get("site_id") or source.get("domain")
        if value:
            return str(value)
    return None


def _train_fingerprints(records: Iterable[ExampleRecord]) -> dict[str, set[str]]:
    result = {
        "ids": set(),
        "split_group_ids": set(),
        "prompts": set(),
        "openuis": set(),
        "structures": set(),
        "pairs": set(),
        "design_mds": set(),
    }
    for record in records:
        result["ids"].add(record.id)
        group = (record.meta or {}).get("split_group_id")
        if group:
            result["split_group_ids"].add(str(group))
        result["prompts"].add(fingerprint_prompt(record.prompt))
        result["openuis"].add(fingerprint_openui(record.openui))
        result["structures"].add(fingerprint_openui_structure(record.openui))
        result["pairs"].add(fingerprint_pair(record.prompt, record.openui))
    return result


def train_generalization_profile(records: Iterable[ExampleRecord]) -> dict[str, Any]:
    rows = list(records)
    pairs: set[tuple[str, ...]] = set()
    triples: set[tuple[str, ...]] = set()
    edits: set[tuple[str, ...]] = set()
    domains: set[str] = set()
    contracts: set[str] = set()
    components: set[str] = set()
    for record in rows:
        components.update(_COMPONENT_RE.findall(record.openui))
        pairs |= _component_combinations(record.openui, 2)
        triples |= _component_combinations(record.openui, 3)
        if composition := _edit_composition(record):
            edits.add(composition)
        if domain := _domain(record):
            domains.add(domain)
        if contract := (record.meta or {}).get("contract_id"):
            contracts.add(str(contract))
    return {
        "component_families": sorted(components),
        "component_pairs": sorted([list(value) for value in pairs]),
        "component_triples": sorted([list(value) for value in triples]),
        "max_tree_depth": max((_program_depth(row.openui) for row in rows), default=0),
        "max_target_tokens": max(
            (len(tokenize_text(row.openui)) for row in rows), default=0
        ),
        "edit_compositions": sorted([list(value) for value in edits]),
        "domains_or_sites": sorted(domains),
        "contract_ids": sorted(contracts),
    }


def generalization_report(
    train_records: Iterable[ExampleRecord],
    held_records: Iterable[ExampleRecord],
) -> dict[str, Any]:
    train = list(train_records)
    held = sorted(held_records, key=lambda record: record.id)
    profile = train_generalization_profile(train)
    fingerprints = _train_fingerprints(train)
    known_pairs = {tuple(value) for value in profile["component_pairs"]}
    known_triples = {tuple(value) for value in profile["component_triples"]}
    known_edits = {tuple(value) for value in profile["edit_compositions"]}
    known_domains = set(profile["domains_or_sites"])
    known_contracts = set(profile["contract_ids"])
    contaminated: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}

    for record in held:
        reasons = find_leakage(record, fingerprints)
        if reasons:
            contaminated.append({"id": record.id, "reasons": reasons})
            continue
        slices: list[str] = []
        if set(_COMPONENT_RE.findall(record.openui)) - set(
            profile["component_families"]
        ):
            slices.append("unseen_component_family")
        if _component_combinations(record.openui, 2) - known_pairs:
            slices.append("unseen_component_pair")
        if _component_combinations(record.openui, 3) - known_triples:
            slices.append("unseen_component_triple")
        if _program_depth(record.openui) > profile["max_tree_depth"]:
            slices.append("deeper_tree")
        if len(tokenize_text(record.openui)) > profile["max_target_tokens"]:
            slices.append("longer_program")
        composition = _edit_composition(record)
        if composition and composition not in known_edits:
            slices.append("new_edit_composition")
        domain = _domain(record)
        if domain and domain not in known_domains:
            slices.append("new_domain_or_site")
        contract = (record.meta or {}).get("contract_id")
        if contract and str(contract) not in known_contracts:
            slices.append("new_contract_version")
        for name in slices:
            counts[name] = counts.get(name, 0) + 1
        rows.append({"id": record.id, "slices": slices})

    return {
        "decontaminated": not contaminated,
        "train_profile": profile,
        "held_out_n": len(held),
        "accepted_n": len(rows),
        "contaminated": contaminated,
        "slice_counts": dict(sorted(counts.items())),
        "records": rows,
        "schema_transfer": schema_transfer_report(train, held),
    }
