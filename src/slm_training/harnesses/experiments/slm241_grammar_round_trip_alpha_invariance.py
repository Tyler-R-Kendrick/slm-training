"""SLM-241 (GRT0-01): D2 canonicalizer round-trip / alpha-invariance stress probe.

``slm_training.dsl.canonicalize.canonicalize`` claims (in its own module
docstring) to be a "confluent codec round-trip canonicalizer": parse to the
grammar-native production stream (``production_codec.encode_openui``),
deterministically re-emit (``decode_productions``), and the result is a
*normal form* -- idempotent, always re-validates, and alpha-invariant (two
programs that differ only in local ``v0, v1, ...``-style binder identifiers
canonicalize to the identical string). This normal-form property is what
backs C3 macro induction, canonical exact match, and "canonical dedup"
callers across the repo (``harnesses/experiments/test_canonical_ast_dedup``
and friends).

The only place this claim was previously exercised is
``tests/test_dsl/test_canonicalize.py`` -- three hand-picked programs plus
one regression fixture for a specific placeholder-aliasing bug. No harness
had asked the question at scale, across the full pinned OpenUI component
schema and the depth/width/viewport/content-class grid the repo's own
coverage-guided typed generator (``data/progspec/generate.py``) already
walks for training-data root construction.

This harness asks a narrow, falsifiable, CPU-only question: across a broad,
generator-produced corpus of real (Lark-grammar-valid) OpenUI programs, does
the canonicalizer's normal-form claim hold -- (1) idempotency
(``canonicalize(canonicalize(x)) == canonicalize(x)``), (2) always-valid
output (``canonicalize(x)`` re-validates through the real parser), and (3)
alpha-invariance (renaming every non-root local binder identifier to a fresh,
disjoint name never changes the canonical form)?

No new gate is implemented and no existing canonicalizer, parser, or
production-codec behavior is changed. This only exercises the real,
unmodified ``slm_training.dsl.canonicalize.canonicalize``,
``slm_training.dsl.parser.validate``, and the typed corpus generator's
candidate-selection/serialization machinery
(``slm_training.data.progspec.generate.ProgramGenerator``) against
synthetic, grammar-validated OpenUI programs.

**Environment note (not a hypothesis, a fixture-construction caveat):** the
canonical entry point for this generator, ``generate_program_specs`` /
``ProgramGenerator.generate_one``, additionally round-trips every candidate
through ``verify_record``'s G2 (schema) gate, which calls
``slm_training.dsl.lang_core.validate`` -- a *bridge-only* function that
raises unconditionally when the official ``@openuidev/lang-core`` Node
bridge is unavailable (it is unavailable in this sandbox; ``uv run python -c
"from slm_training.dsl import lang_core; lang_core.bridge_available()"``
returns ``False``). That is a separate, pre-existing environment gap in the
generator's verification path, not this harness's subject, so this harness
calls the generator's public ``ProgramGenerator._choose`` /
``ProgramGenerator._build_program`` candidate machinery directly (the same
coverage-driven candidate selection and typed serialization
``generate_one`` uses) and validates each candidate through
``slm_training.dsl.parser.validate`` -- the hybrid backend, which correctly
falls back to the in-process Lark grammar when the bridge is absent -- the
same backend the canonicalizer itself calls.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from slm_training.data.progspec.generate import GeneratorConfig, ProgramGenerator
from slm_training.dsl.canonicalize import canonicalize
from slm_training.dsl.lang_core import ParseError
from slm_training.dsl.parser import validate
from slm_training.versioning import build_version_stamp

__all__ = [
    "EXPERIMENT_ID",
    "MATRIX_SET",
    "MATRIX_VERSION",
    "DEFAULT_SEEDS",
    "DEFAULT_COUNT_PER_SEED",
    "CandidateRow",
    "GrammarRoundTripReport",
    "generate_candidate_sources",
    "mask_literals",
    "unmask_literals",
    "binder_names",
    "permute_binders",
    "run_round_trip_fixture",
    "render_markdown",
]

MATRIX_VERSION = "grt0-01-v1"
MATRIX_SET = "slm241_grammar_round_trip_alpha_invariance"
EXPERIMENT_ID = "slm241-grammar-round-trip-alpha-invariance"

DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2)
DEFAULT_COUNT_PER_SEED = 50

_HYPOTHESIS = (
    "The D2 canonicalizer (slm_training.dsl.canonicalize.canonicalize, built "
    "on production_codec.encode_openui/decode_productions) is a stable "
    "normal form across a broad, coverage-guided corpus of generated OpenUI "
    "programs -- not just the 3 hand-picked examples in "
    "tests/test_dsl/test_canonicalize.py: (1) canonicalize is idempotent, "
    "(2) canonicalize's output always re-validates through the real parser, "
    "and (3) renaming every non-root local binder identifier to a fresh, "
    "disjoint set of names never changes the canonical form (alpha-"
    "invariance)."
)

_FALSIFIER = (
    "Any generated candidate for which canonicalize(canonicalize(x)) != "
    "canonicalize(x); or canonicalize(x) fails to re-validate through "
    "slm_training.dsl.parser.validate; or a binder-permuted variant of x "
    "(same layout, only non-root local identifiers renamed) is grammar-"
    "valid but canonicalizes to a different string than x."
)

_HONEST_CAVEATS = (
    "Fixture/wiring evidence only: no checkpoint, GPU run, or ship-gate "
    "claim is made or implied.",
    "This is a positive/ceiling-style probe: it asks whether an existing, "
    "documented normal-form claim holds at generator scale, not whether "
    "some new mechanism should ship. It does not change canonicalize, "
    "production_codec, the parser, or any generator default.",
    "The corpus generator (ProgramGenerator._choose / _build_program) is "
    "the same coverage-guided candidate machinery generate_one() uses, but "
    "this harness calls it directly and validates candidates through "
    "slm_training.dsl.parser.validate (hybrid, Lark-fallback) instead of "
    "generate_one()'s verify_record path, because verify_record's G2 gate "
    "calls the bridge-only slm_training.dsl.lang_core.validate, which "
    "raises unconditionally when the official @openuidev/lang-core Node "
    "bridge is unavailable -- as it is in this sandbox. Cross-backend "
    "parity between the official lang-core parser and the in-process Lark "
    "grammar is therefore untested here; only the Lark-backed path is "
    "exercised.",
    "The binder renamer is a regex-based identifier substitution (string "
    "literals are masked first, then whole-word non-root binder "
    "identifiers are substituted via a single alternation regex, then "
    "literals are restored) -- not an AST-level rename. It is exercised "
    "against, and passes, the same placeholder-aliasing edge case already "
    "regression-tested in tests/test_dsl/test_canonicalize.py (binder "
    "stems appearing inside quoted placeholder text), but it is a "
    "test-harness utility, not a reusable production renamer.",
    "Candidates with 0 or 1 non-root binder (root refers to nothing else, "
    "or exactly one other statement) have no possible nontrivial "
    "permutation and are recorded as trivial (alpha_invariant=None), not "
    "counted toward the alpha-invariance claim.",
)

_LIT_RE = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')
_LIT_TOKEN_RE = re.compile(r"\x00LIT(\d+)\x00")
_BINDER_LHS_RE = re.compile(r"(?m)^([a-z_][A-Za-z0-9_]*)\s*=")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _digest(value: Any) -> str:
    return _sha256(_canonical_json(value))


def mask_literals(source: str) -> tuple[str, list[str]]:
    """Replace every quoted string literal with a positional token.

    Protects literal text (including placeholder-shaped substrings like
    ``:form.title``) from the binder-identifier substitution below.
    """
    literals: list[str] = []

    def repl(match: re.Match[str]) -> str:
        literals.append(match.group(0))
        return f"\x00LIT{len(literals) - 1}\x00"

    return _LIT_RE.sub(repl, source), literals


def unmask_literals(masked: str, literals: list[str]) -> str:
    return _LIT_TOKEN_RE.sub(lambda m: literals[int(m.group(1))], masked)


def binder_names(source: str) -> list[str]:
    """Ordered, de-duplicated statement-LHS binder identifiers (includes ``root``)."""
    seen: list[str] = []
    for match in _BINDER_LHS_RE.finditer(source):
        name = match.group(1)
        if name not in seen:
            seen.append(name)
    return seen


def permute_binders(source: str) -> tuple[str | None, dict[str, str]]:
    """Rename every non-``root`` binder identifier to a fresh, disjoint name.

    ``root`` is never renamed -- the grammar requires exactly one ``root``
    binding (see production_codec's "missing root binding" checks). Returns
    ``(None, {})`` when fewer than 2 non-root binders exist (no nontrivial
    permutation is possible). The rename is a cyclic rotation by 1 over a
    guaranteed-fresh ``zzalt{i}`` name pool, so every renamed binder gets a
    genuinely different identifier and the pool cannot collide with any
    generator-produced stem (component-name-derived, never ``zzalt``-prefixed).
    """
    names = [name for name in binder_names(source) if name != "root"]
    if len(names) < 2:
        return None, {}
    pool = [f"zzalt{i}" for i in range(len(names))]
    mapping = {old: pool[(i + 1) % len(names)] for i, old in enumerate(names)}
    masked, literals = mask_literals(source)
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True)) + r")\b"
    )
    renamed_masked = pattern.sub(lambda m: mapping[m.group(1)], masked)
    return unmask_literals(renamed_masked, literals), mapping


@dataclass(frozen=True)
class CandidateRow:
    """One generated candidate's round-trip / alpha-invariance outcome."""

    seed: int
    index: int
    components: tuple[str, ...]
    depth: int
    width: int
    binder_count: int
    source_len: int
    idempotent: bool
    revalidates: bool
    revalidate_error: str
    alpha_invariant: bool | None
    permuted_valid: bool | None
    permute_error: str
    canonicalize_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "index": self.index,
            "components": list(self.components),
            "depth": self.depth,
            "width": self.width,
            "binder_count": self.binder_count,
            "source_len": self.source_len,
            "idempotent": self.idempotent,
            "revalidates": self.revalidates,
            "revalidate_error": self.revalidate_error,
            "alpha_invariant": self.alpha_invariant,
            "permuted_valid": self.permuted_valid,
            "permute_error": self.permute_error,
            "canonicalize_ms": self.canonicalize_ms,
        }


def generate_candidate_sources(
    seed: int, count: int
) -> list[tuple[str, tuple[str, ...], int, int]]:
    """``count`` real, Lark-grammar-valid OpenUI sources via the canonical
    typed generator's own candidate-selection + serialization machinery.

    Reuses ``ProgramGenerator._choose`` (coverage-driven candidate pick over
    the pinned OpenUI schema) and ``ProgramGenerator._build_program`` (typed
    AST -> source serialize) directly -- the exact primitives
    ``generate_one`` composes -- without going through ``generate_one``'s
    bridge-only ``verify_record`` call (see module docstring).
    """
    generator = ProgramGenerator(GeneratorConfig(), seed=seed)
    out: list[tuple[str, tuple[str, ...], int, int]] = []
    for _ in range(count):
        try:
            candidate = generator._choose()
        except ValueError:
            break  # candidate grid exhausted for this seed/config
        source, _cells = generator._build_program(candidate)
        out.append((source, candidate.components, candidate.depth, candidate.width))
    return out


def _row_for(seed: int, index: int, source: str, components: tuple[str, ...], depth: int, width: int) -> CandidateRow:
    validate(source)  # fail closed: only real grammar-valid candidates are scored
    t0 = time.perf_counter()
    c1 = canonicalize(source)
    c2 = canonicalize(c1)
    canonicalize_ms = (time.perf_counter() - t0) * 1000.0
    idempotent = c1 == c2

    revalidates = True
    revalidate_error = ""
    try:
        validate(c1)
    except (ParseError, ValueError) as exc:
        revalidates = False
        revalidate_error = str(exc).splitlines()[0][:200]

    renamed, mapping = permute_binders(source)
    alpha_invariant: bool | None = None
    permuted_valid: bool | None = None
    permute_error = ""
    if renamed is not None:
        try:
            validate(renamed)
            permuted_valid = True
        except (ParseError, ValueError) as exc:
            permuted_valid = False
            permute_error = str(exc).splitlines()[0][:200]
        if permuted_valid:
            alpha_invariant = canonicalize(renamed) == c1

    return CandidateRow(
        seed=seed,
        index=index,
        components=components,
        depth=depth,
        width=width,
        binder_count=len(mapping),
        source_len=len(source),
        idempotent=idempotent,
        revalidates=revalidates,
        revalidate_error=revalidate_error,
        alpha_invariant=alpha_invariant,
        permuted_valid=permuted_valid,
        permute_error=permute_error,
        canonicalize_ms=canonicalize_ms,
    )


def _resolve_disposition(rows: list[CandidateRow]) -> tuple[str, str]:
    if not rows:
        return "inconclusive", "No candidates were generated."
    idempotency_failures = [r for r in rows if not r.idempotent]
    revalidate_failures = [r for r in rows if not r.revalidates]
    nontrivial = [r for r in rows if r.alpha_invariant is not None]
    alpha_failures = [r for r in nontrivial if not r.alpha_invariant]
    permute_invalid = [r for r in rows if r.permuted_valid is False]

    if not nontrivial:
        return (
            "inconclusive",
            "No candidate had 2+ non-root binders, so the alpha-invariance "
            "arm of the hypothesis was never exercised.",
        )

    if idempotency_failures or revalidate_failures or alpha_failures:
        return (
            "gap_confirmed",
            f"{len(idempotency_failures)}/{len(rows)} candidates were not "
            f"idempotent, {len(revalidate_failures)}/{len(rows)} did not "
            f"re-validate, and {len(alpha_failures)}/{len(nontrivial)} "
            "non-trivial candidates violated alpha-invariance -- the "
            "canonicalizer's normal-form claim does not hold universally "
            "at this generator scale.",
        )

    return (
        "ceiling_confirmed_at_scale",
        f"All {len(rows)} generated candidates ({len(nontrivial)} with 2+ "
        "non-root binders) were idempotent under canonicalize, always "
        "re-validated, and were alpha-invariant under a full non-root "
        f"binder permutation ({len(permute_invalid)} permuted variants "
        "failed to even parse, which would itself be a renamer artifact, "
        "not counted against the claim). The canonicalizer's documented "
        "normal-form property holds across this generator-scale corpus, "
        "beyond the 3 hand-picked unit-test examples.",
    )


@dataclass(frozen=True)
class GrammarRoundTripReport:
    """Full fixture report for SLM-241."""

    schema: str = "GrammarRoundTripReportV1"
    matrix_set: str = MATRIX_SET
    matrix_version: str = MATRIX_VERSION
    experiment_id: str = EXPERIMENT_ID
    run_id: str = EXPERIMENT_ID
    status: str = "fixture"
    claim_class: str = "wiring"
    hypothesis: str = _HYPOTHESIS
    falsifier: str = _FALSIFIER
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    count_per_seed: int = DEFAULT_COUNT_PER_SEED
    rows: tuple[CandidateRow, ...] = field(default_factory=tuple)
    gate_hash: str = ""
    disposition: str = "inconclusive"
    disposition_rationale: str = ""
    honest_caveats: tuple[str, ...] = _HONEST_CAVEATS
    version_stamp: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "matrix_set": self.matrix_set,
            "matrix_version": self.matrix_version,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "status": self.status,
            "claim_class": self.claim_class,
            "hypothesis": self.hypothesis,
            "falsifier": self.falsifier,
            "seeds": list(self.seeds),
            "count_per_seed": self.count_per_seed,
            "rows": [r.to_dict() for r in self.rows],
            "gate_hash": self.gate_hash,
            "disposition": self.disposition,
            "disposition_rationale": self.disposition_rationale,
            "honest_caveats": list(self.honest_caveats),
            "version_stamp": dict(self.version_stamp),
            "timestamp": self.timestamp,
        }

    def to_json(self, path) -> None:  # pragma: no cover - thin IO helper
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GrammarRoundTripReport":
        rows = tuple(
            CandidateRow(
                seed=int(r["seed"]),
                index=int(r["index"]),
                components=tuple(str(c) for c in r.get("components", ())),
                depth=int(r["depth"]),
                width=int(r["width"]),
                binder_count=int(r["binder_count"]),
                source_len=int(r["source_len"]),
                idempotent=bool(r["idempotent"]),
                revalidates=bool(r["revalidates"]),
                revalidate_error=str(r.get("revalidate_error", "")),
                alpha_invariant=(
                    None if r.get("alpha_invariant") is None else bool(r["alpha_invariant"])
                ),
                permuted_valid=(
                    None if r.get("permuted_valid") is None else bool(r["permuted_valid"])
                ),
                permute_error=str(r.get("permute_error", "")),
                canonicalize_ms=float(r["canonicalize_ms"]),
            )
            for r in data.get("rows", ())
        )
        return cls(
            schema=str(data.get("schema", "GrammarRoundTripReportV1")),
            matrix_set=str(data.get("matrix_set", MATRIX_SET)),
            matrix_version=str(data.get("matrix_version", MATRIX_VERSION)),
            experiment_id=str(data.get("experiment_id", EXPERIMENT_ID)),
            run_id=str(data.get("run_id", EXPERIMENT_ID)),
            status=str(data.get("status", "fixture")),
            claim_class=str(data.get("claim_class", "wiring")),
            hypothesis=str(data.get("hypothesis", _HYPOTHESIS)),
            falsifier=str(data.get("falsifier", _FALSIFIER)),
            seeds=tuple(int(s) for s in data.get("seeds", DEFAULT_SEEDS)),
            count_per_seed=int(data.get("count_per_seed", DEFAULT_COUNT_PER_SEED)),
            rows=rows,
            gate_hash=str(data.get("gate_hash", "")),
            disposition=str(data.get("disposition", "inconclusive")),
            disposition_rationale=str(data.get("disposition_rationale", "")),
            honest_caveats=tuple(data.get("honest_caveats", _HONEST_CAVEATS)),
            version_stamp=dict(data.get("version_stamp", {})),
            timestamp=str(data.get("timestamp", _now())),
        )


def run_round_trip_fixture(
    *,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    count_per_seed: int = DEFAULT_COUNT_PER_SEED,
    run_id: str | None = None,
) -> GrammarRoundTripReport:
    """Generate candidates for every seed and score idempotency / re-validation
    / alpha-invariance through the real, unmodified canonicalizer + parser."""
    rows: list[CandidateRow] = []
    for seed in seeds:
        candidates = generate_candidate_sources(seed, count_per_seed)
        for index, (source, components, depth, width) in enumerate(candidates):
            rows.append(_row_for(seed, index, source, components, depth, width))

    disposition, rationale = _resolve_disposition(rows)

    # canonicalize_ms is wall-clock timing noise, not a claim outcome -- excluded
    # from the gate hash so re-running the same seeds/counts is deterministic.
    def _stable_row_dict(row: CandidateRow) -> dict[str, Any]:
        data = row.to_dict()
        data.pop("canonicalize_ms", None)
        return data

    payload = {"row_digests": sorted(_digest(_stable_row_dict(r)) for r in rows)}
    gate_hash = _sha256(_canonical_json(payload))

    return GrammarRoundTripReport(
        run_id=run_id or f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
        seeds=seeds,
        count_per_seed=count_per_seed,
        rows=tuple(rows),
        gate_hash=gate_hash,
        disposition=disposition,
        disposition_rationale=rationale,
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm241_grammar_round_trip_alpha_invariance",
        ),
    )


def render_markdown(report: GrammarRoundTripReport) -> str:
    nontrivial = [r for r in report.rows if r.alpha_invariant is not None]
    trivial = len(report.rows) - len(nontrivial)
    avg_ms = (
        sum(r.canonicalize_ms for r in report.rows) / len(report.rows)
        if report.rows
        else 0.0
    )
    lines = [
        f"# SLM-241 (GRT0-01): D2 canonicalizer round-trip / alpha-invariance stress probe ({report.run_id})",
        "",
        f"**Matrix set:** `{report.matrix_set}`",
        f"**Version:** `{report.matrix_version}`",
        f"**Status:** {report.status}",
        f"**Claim class:** {report.claim_class}",
        f"**Gate hash:** `{report.gate_hash[:16]}...`",
        f"**Disposition:** {report.disposition} — {report.disposition_rationale}",
        "",
        "## Hypothesis",
        "",
        report.hypothesis,
        "",
        "## Falsifier",
        "",
        report.falsifier,
        "",
        "## Honest caveats",
        "",
        *(f"- {c}" for c in report.honest_caveats),
        "",
        "## Recipe",
        "",
        f"- seeds: `{list(report.seeds)}`, count per seed: `{report.count_per_seed}`",
        f"- candidates scored: `{len(report.rows)}` (`{len(nontrivial)}` with 2+ non-root binders, `{trivial}` trivial)",
        f"- mean `canonicalize()`+`canonicalize(canonicalize())` latency: `{avg_ms:.3f} ms`",
        "- backend: hybrid OpenUI (Lark fallback — the official lang-core Node bridge is unavailable in this sandbox)",
        "",
        "## Per-seed summary",
        "",
        "| seed | candidates | idempotent | revalidates | alpha-invariant (of non-trivial) | permuted-invalid |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for seed in report.seeds:
        seed_rows = [r for r in report.rows if r.seed == seed]
        seed_nontrivial = [r for r in seed_rows if r.alpha_invariant is not None]
        idem = sum(1 for r in seed_rows if r.idempotent)
        reval = sum(1 for r in seed_rows if r.revalidates)
        alpha_ok = sum(1 for r in seed_nontrivial if r.alpha_invariant)
        perm_invalid = sum(1 for r in seed_rows if r.permuted_valid is False)
        lines.append(
            f"| {seed} | {len(seed_rows)} | {idem}/{len(seed_rows)} | {reval}/{len(seed_rows)} | "
            f"{alpha_ok}/{len(seed_nontrivial)} | {perm_invalid} |"
        )

    counterexamples = [
        r
        for r in report.rows
        if not r.idempotent or not r.revalidates or r.alpha_invariant is False
    ]
    lines += ["", "## Counterexamples (if any)", ""]
    if counterexamples:
        lines.append("| seed | index | components | depth | width | idempotent | revalidates | alpha-invariant | error |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for r in counterexamples:
            err = r.revalidate_error or r.permute_error or "—"
            lines.append(
                f"| {r.seed} | {r.index} | {', '.join(r.components)} | {r.depth} | {r.width} | "
                f"{r.idempotent} | {r.revalidates} | {r.alpha_invariant} | {err} |"
            )
    else:
        lines.append("None — every scored candidate was idempotent, re-validated, and alpha-invariant.")

    lines += [
        "",
        "## No-go for promotion",
        "",
        "This report is wiring/fixture evidence only. It does not change "
        "`canonicalize`, `production_codec`, `slm_training.dsl.parser`, or "
        "any generator default, does not train a model, and makes no ship "
        "or gate claim. It scales the D2 canonicalizer's documented "
        "normal-form claim from 3 hand-picked unit-test examples to a "
        "generator-driven corpus and records the outcome honestly, whether "
        "confirming or falsifying.",
        "",
        "## Reproducibility",
        "",
        "```bash",
        "python -m scripts.run_slm241_grammar_round_trip_alpha_invariance --mode plan-only",
        "python -m scripts.run_slm241_grammar_round_trip_alpha_invariance --mode fixture",
        "```",
        "",
    ]
    return "\n".join(lines)
