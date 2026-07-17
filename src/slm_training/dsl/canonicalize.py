"""D2 canonicalizer: a semantics-preserving canonical form for OpenUI ASTs.

Two OpenUI programs that differ only in binder names, statement order, or style
literals denote the same layout. Canonicalization maps each equivalence class to
one representative string, so ``canonical exact match`` and repeated-subterm
detection (C3 macro induction) work on meaning rather than surface form.

**What this is.** A *confluent codec round-trip* canonicalizer: parse to the
grammar-native production stream (`production_codec.encode_openui`) and
deterministically re-emit (`decode_productions`). The codec already fixes a
single canonical statement order (topological, first-use), renames binders to a
De Bruijn-style `v0, v1, …` pool, and strips style literals — so the round-trip
is a normal form by construction. The result is validated back through the
official parser, and the transform is idempotent.

**What this is not.** Not an e-graph / equality-saturation engine, and not a
semantic simplifier: it does not elide schema defaults or flatten containers
(those rewrites risk changing meaning and are left to a future, schema-checked
pass). Naming it a "canonicalizer" refers only to the normal-form property above.

**Known caveat (arXiv:2401.02948).** The binder renaming is context-*insensitive*
De Bruijn-style: it canonicalizes alpha-equivalent whole programs, but it does
not by itself detect all context-sensitive common subterms. C3 macro induction
must not assume canonical binder identity implies shared subterm context.
"""

from __future__ import annotations

import hashlib

from slm_training.dsl.production_codec import decode_productions, encode_openui


def canonicalize(source: str, *, dsl: str | None = None, validate: bool = True) -> str:
    """Return the canonical OpenUI string for ``source``.

    ``validate`` (default) round-trips the result through the official parser and
    raises if it does not parse — canonicalization must never emit an invalid
    program. Set ``validate=False`` only in hot paths where the input is already
    trusted valid.
    """
    program = encode_openui(source, dsl=dsl)
    canonical = decode_productions(program.tokens, program.slot_contract)
    if validate:
        from slm_training.dsl.parser import validate as _validate

        _validate(canonical, dsl=dsl)
    return canonical


def canonical_fingerprint(source: str, *, dsl: str | None = None) -> str:
    """Stable sha256 of the canonical form — one hash per equivalence class."""
    return hashlib.sha256(
        canonicalize(source, dsl=dsl, validate=False).encode("utf-8")
    ).hexdigest()


def canonical_equal(a: str, b: str, *, dsl: str | None = None) -> bool:
    """Whether ``a`` and ``b`` denote the same layout (canonical exact match)."""
    try:
        return canonicalize(a, dsl=dsl, validate=False) == canonicalize(
            b, dsl=dsl, validate=False
        )
    except Exception:  # noqa: BLE001 - an unparseable side is never "equal"
        return False
