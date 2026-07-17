"""E1 bits-per-semantic-decision: how much grammar must a model learn?

Formalizes the open question "how small can the model be if the grammar is
externalized" (a confirmed literature gap; TinyStories is the nearest empirical
anchor, no paper claims this framing). The idea: measure a corpus's information
content over **grammar choice points only**, tokenizer-independently, and report
how many model parameters are spent per bit of that content.

Two token streams over the same programs:

* **production** — the `ProductionCodec` stream (`dsl/production_codec.encode_openui`),
  where non-lexical punctuation/structure is already externalized, so every token
  is a genuine grammar decision (which production, which slot, which reference);
* **surface** — the compiler-derived lexeme stream (`dsl/parser.lexical_tokens`),
  which still carries structural symbols.

For a stream we compute the corpus unigram description length
``total_bits = sum_tokens -log2 p_hat(token)`` (= ``N * H`` of the empirical
distribution). ``bits_per_decision`` is the entropy ``H`` in bits; it is the
per-decision target a model must reproduce. ``params_per_bit(n_params)`` divides
a model's trainable parameter count by the corpus's total choice bits — the
capacity spent per bit of externalized grammar, the number the capacity ladder
(B3) compares across representations.

Diagnostic/measurement only; no ship claim.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from slm_training.dsl.production_codec import (
    BUILTIN_PREFIX,
    DIR_PREFIX,
    LIT_PREFIX,
    NAME_PREFIX,
    OPEN_PREFIX,
    PUNCT_PREFIX,
    REF_PREFIX,
    SLOT_PREFIX,
    STATE_REF_PREFIX,
    encode_openui,
    to_choice_stream,
)

# Token first-character → semantic category, for a transparent breakdown.
# Order matters: STATE_REF_PREFIX ("$@") is checked before SLOT_PREFIX ("@").
_CATEGORY_PREFIXES: tuple[tuple[str, str], ...] = (
    (STATE_REF_PREFIX, "state_ref"),
    (OPEN_PREFIX, "production"),
    (DIR_PREFIX, "direction"),
    (SLOT_PREFIX, "slot"),
    (REF_PREFIX, "reference"),
    (LIT_PREFIX, "literal"),
    (BUILTIN_PREFIX, "builtin"),
    (NAME_PREFIX, "name"),
    (PUNCT_PREFIX, "punct"),
)


def categorize(token: str) -> str:
    """Classify a production-codec token into a semantic-decision category."""
    for prefix, name in _CATEGORY_PREFIXES:
        if token.startswith(prefix):
            return name
    return "structural"


@dataclass(frozen=True)
class SemanticBitsConfig:
    # Count slot-pointer tokens ("@i") as decisions. Slots are genuine filler
    # choices; excluding them isolates pure structural decisions.
    include_slots: bool = True
    # Count literal tokens ("#...") as decisions.
    include_literals: bool = True


def _openui_of(record: Any) -> tuple[str, list[str] | None]:
    if isinstance(record, str):
        return record, None
    openui = getattr(record, "openui", None)
    placeholders = getattr(record, "placeholders", None)
    if openui is None and isinstance(record, dict):
        openui = record.get("openui")
        placeholders = record.get("placeholders")
    return str(openui or ""), (list(placeholders) if placeholders else None)


def _production_tokens(
    record: Any, cfg: SemanticBitsConfig, *, choice: bool = False
) -> list[str]:
    source, placeholders = _openui_of(record)
    if not source.strip():
        return []
    try:
        program = encode_openui(source, slot_contract=placeholders)
    except Exception:  # noqa: BLE001 - codec-unencodable contributes nothing
        # Mirrors the surface stream's policy. The skip is visible, never
        # silent: n_scored_programs < n_programs in the report (e.g. rootless
        # language-contract records the document codec cannot encode).
        return []
    # B1 choice stream: elide grammar-forced framing BEFORE category filtering
    # (the transform needs the framing intact to locate statement boundaries).
    stream = to_choice_stream(program.tokens) if choice else program.tokens
    tokens: list[str] = []
    for token in stream:
        cat = categorize(token)
        if cat == "slot" and not cfg.include_slots:
            continue
        if cat == "literal" and not cfg.include_literals:
            continue
        tokens.append(token)
    return tokens


def _surface_tokens(record: Any) -> list[str]:
    from slm_training.dsl.parser import lexical_tokens

    source, _ = _openui_of(record)
    if not source.strip():
        return []
    try:
        return list(lexical_tokens(source))
    except Exception:  # noqa: BLE001 - unparseable surface contributes nothing
        return []


def _description_length(tokens: list[str]) -> dict[str, Any]:
    """Corpus unigram bits: total = sum -log2 p_hat; per-token = entropy H."""
    counts = Counter(tokens)
    n = sum(counts.values())
    if n == 0:
        return {
            "n_decisions": 0,
            "alphabet_size": 0,
            "total_bits": 0.0,
            "bits_per_decision": None,
        }
    total_bits = 0.0
    for token, count in counts.items():
        p = count / n
        total_bits += -count * math.log2(p)
    return {
        "n_decisions": n,
        "alphabet_size": len(counts),
        "total_bits": total_bits,
        "bits_per_decision": total_bits / n,
    }


def semantic_bits(
    records: Iterable[Any],
    *,
    stream: str = "production",
    config: SemanticBitsConfig | None = None,
    params: int | None = None,
) -> dict[str, Any]:
    """Bits-per-decision for one representation over a corpus.

    ``stream`` is ``"production"`` (grammar choice points), ``"choice"`` (the B1
    choice-sequence stream — production minus grammar-forced framing), or
    ``"surface"`` (compiler lexemes). ``params`` (a model's trainable parameter
    count) adds a ``params_per_bit`` field.
    """
    cfg = config or SemanticBitsConfig()
    records = list(records)
    if stream == "production":
        per_program = [_production_tokens(r, cfg) for r in records]
    elif stream == "choice":
        per_program = [_production_tokens(r, cfg, choice=True) for r in records]
    elif stream == "surface":
        per_program = [_surface_tokens(r) for r in records]
    else:  # pragma: no cover - guarded by callers/tests
        raise ValueError(f"unknown stream {stream!r}")

    flat = [tok for program in per_program for tok in program]
    report = _description_length(flat)
    scored = [p for p in per_program if p]
    report.update(
        {
            "stream": stream,
            "n_programs": len(records),
            "n_scored_programs": len(scored),
            "mean_decisions_per_program": (
                report["n_decisions"] / len(scored) if scored else None
            ),
        }
    )
    if stream in ("production", "choice"):
        cats = Counter(categorize(tok) for tok in flat)
        report["by_category"] = dict(sorted(cats.items()))
    if params is not None and report["total_bits"] > 0:
        report["params"] = int(params)
        report["params_per_bit"] = int(params) / report["total_bits"]
    return report


def compare_representations(
    records: Iterable[Any],
    *,
    config: SemanticBitsConfig | None = None,
    params: int | None = None,
) -> dict[str, Any]:
    """Production vs choice vs surface bits, and the externalization ratios."""
    records = list(records)
    production = semantic_bits(
        records, stream="production", config=config, params=params
    )
    choice = semantic_bits(records, stream="choice", config=config, params=params)
    surface = semantic_bits(records, stream="surface", config=config, params=params)
    prod_bits = production["total_bits"]
    choice_bits = choice["total_bits"]
    surf_bits = surface["total_bits"]
    # Ratios are only meaningful over the SAME scored programs. When a stream
    # skips records the others score (e.g. rootless language-contract records
    # the document codec cannot encode), cross-stream ratios would compare
    # different corpora — report None and flag the mismatch instead.
    scored = {
        production["n_scored_programs"],
        choice["n_scored_programs"],
        surface["n_scored_programs"],
    }
    mismatch = len(scored) > 1
    return {
        "production": production,
        "choice": choice,
        "surface": surface,
        "scored_programs_mismatch": mismatch,
        # >1 means externalizing the grammar shrinks the corpus's total choice
        # bits (fewer bits for the model to learn) relative to raw surface.
        "surface_to_production_bit_ratio": (
            surf_bits / prod_bits if prod_bits > 0 and not mismatch else None
        ),
        "surface_to_choice_bit_ratio": (
            surf_bits / choice_bits if choice_bits > 0 and not mismatch else None
        ),
        "decision_reduction_ratio": (
            surface["n_decisions"] / production["n_decisions"]
            if production["n_decisions"] > 0 and not mismatch
            else None
        ),
    }
