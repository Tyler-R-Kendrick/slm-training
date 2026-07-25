"""Partial-state classification for OpenUI CAP0 partial programs.

SLM-359 (DSH1-07). Every candidate partial source is classified through the
declared grammar capability adapter (DSH1-01
:class:`GrammarCapabilityAdapterV1`) into exactly one
:class:`PartialStateClassV1`:

* ``COMPLETE`` — the source parses as a full program under the pack's primary
  document start symbol and passes ``static_validate``;
* ``FRAGMENT(start_symbol)`` — not complete, but parses under another named,
  declared start symbol via the live fragment parser;
* ``PREFIX(frontier)`` — not complete and not a named fragment, but the live
  completion frontier is non-empty **and** a compiler certificate proves at
  least one bounded accepting completion: a declared pack witness whose token
  stream extends the candidate at a lexer boundary, replayed through the
  frontier at every step to a verified terminal (``static_validate``);
* ``INVALID`` — anything else: a cut inside a lexical token or opaque marker,
  an unreachable partial (empty frontier), or a partial with no certified
  bounded completion.

Cut discipline: candidates are only ever accepted at lexer/production
boundaries. A cut inside a string token or an opaque placeholder marker is
``INVALID`` before any parse/frontier probe runs. Arbitrary internal-hole
generation is **excluded** from initial CAP0: this module classifies
compiler-reachable states only and never synthesizes holes.

``UNKNOWN`` / ``PARTIAL`` support coverage is reported on the report but
**never** authorizes destructive pruning: :func:`authorize_pruning` raises
:class:`PartialStatePruneError` unless the class is admitted with ``FULL``
coverage. Stop rule: :func:`family_valid` invalidates a family when any
member partial is unreachable from the live compiler/decoder.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any, Iterable, Literal

from slm_training.dsl.grammar_capabilities import (
    GrammarCapabilityAdapterV1,
    UnsupportedCapabilityV1,
)
from slm_training.dsl.harness_dsl import runtime_symbols_for_payload

SCHEMA_VERSION = "partial_states/v1"
CLASSIFIER_VERSION = "v1"

# Boundary characters that may legally follow a lexer token; a witness
# continuation only certifies a candidate when the next character after the
# candidate cut is one of these (mirrors the incremental-engine trim set).
_BOUNDARY_CHARS = frozenset(" \n\t=,()[]\"'")

# Frontier terminals that carry no program content when measuring singletons.
_FRONTIER_IGNORABLE = frozenset({"$END", "WS_INLINE", "COMMENT", "_NL"})

_STRING_RE = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')
_TRAILING_MARKER_RE = re.compile(r":[A-Za-z0-9_.]*\Z")


class PartialStateClassV1(str, Enum):
    COMPLETE = "COMPLETE"
    FRAGMENT = "FRAGMENT"
    PREFIX = "PREFIX"
    INVALID = "INVALID"


SupportCoverageV1 = Literal["full", "partial", "unknown"]


class PartialStatePruneError(ValueError):
    """Destructive pruning was requested on non-FULL support coverage."""


@dataclass(frozen=True)
class CompletionCertificateV1:
    """Compiler certificate: one verified bounded accepting continuation."""

    witness_source: str
    completion_suffix: str
    completion_cost: int
    replay_steps: int
    terminal_verified: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "witness_source": self.witness_source,
            "completion_suffix": self.completion_suffix,
            "completion_cost": self.completion_cost,
            "replay_steps": self.replay_steps,
            "terminal_verified": self.terminal_verified,
        }


@dataclass(frozen=True)
class PartialStateReportV1:
    """Classification outcome for one candidate partial source."""

    state_class: PartialStateClassV1
    reason: str
    start_symbol: str | None = None
    frontier: tuple[str, ...] = ()
    open_nonterminals: tuple[str, ...] = ()
    required_roles: tuple[str, ...] = ()
    required_binders: tuple[str, ...] = ()
    exact_singleton_positions: tuple[int, ...] = ()
    support_coverage: SupportCoverageV1 = "unknown"
    min_completion_cost: int | None = None
    certificate: CompletionCertificateV1 | None = None
    schema_version: str = SCHEMA_VERSION
    classifier_version: str = CLASSIFIER_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "classifier_version": self.classifier_version,
            "state_class": self.state_class.value,
            "reason": self.reason,
            "start_symbol": self.start_symbol,
            "frontier": list(self.frontier),
            "open_nonterminals": list(self.open_nonterminals),
            "required_roles": list(self.required_roles),
            "required_binders": list(self.required_binders),
            "exact_singleton_positions": list(self.exact_singleton_positions),
            "support_coverage": self.support_coverage,
            "min_completion_cost": self.min_completion_cost,
            "certificate": (
                self.certificate.to_dict() if self.certificate is not None else None
            ),
        }


# --------------------------------------------------------------------------- #
# Cut discipline: never split a lexical token or an opaque marker
# --------------------------------------------------------------------------- #


def _inside_unterminated_string(source: str) -> bool:
    """True when the source ends inside a string token (odd unescaped quote)."""
    for quote in ('"', "'"):
        count = 0
        index = 0
        while index < len(source):
            char = source[index]
            if char == "\\":
                index += 2
                continue
            if char == quote:
                count += 1
            index += 1
        if count % 2 == 1:
            return True
    return False


def _cut_violation(
    source: str, *, placeholder_re: re.Pattern[str] | None
) -> str | None:
    """Return a stable reason when the cut splits a token/marker, else None."""
    if _inside_unterminated_string(source):
        return "cut_inside_token:string"
    if placeholder_re is not None:
        stripped = _STRING_RE.sub('""', source)
        match = _TRAILING_MARKER_RE.search(stripped)
        if match is not None:
            run = match.group(0)
            if run and placeholder_re.fullmatch(run) is None:
                return "cut_inside_marker:placeholder"
    return None


# --------------------------------------------------------------------------- #
# Lexer helpers (boundary replay + completion cost)
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=8)
def _lexer(grammar_path: str, starts: tuple[str, ...]) -> Any:
    from lark import Lark

    with open(grammar_path, encoding="utf-8") as handle:
        return Lark(
            handle.read(),
            start=list(starts),
            parser="lalr",
            maybe_placeholders=False,
        )


def _lex_suffix(grammar_path: str, starts: tuple[str, ...], suffix: str) -> list[Any]:
    """Lex one completion suffix; raises on any non-boundary content."""
    parser = _lexer(grammar_path, starts)
    return list(parser.lex(suffix))


def _grammar_path(adapter: GrammarCapabilityAdapterV1) -> str | None:
    info = getattr(adapter.pack.backend, "info", None)
    path = getattr(info, "grammar_path", None)
    if path is None:
        return None
    return str(path)


# --------------------------------------------------------------------------- #
# Frontier probes
# --------------------------------------------------------------------------- #


def _frontier(
    adapter: GrammarCapabilityAdapterV1, prefix: str
) -> frozenset[str] | None:
    frontier = adapter.completion_frontier(prefix)
    if isinstance(frontier, UnsupportedCapabilityV1):
        return None
    return frozenset(str(item) for item in frontier)


def _open_nonterminals(
    adapter: GrammarCapabilityAdapterV1, frontier: frozenset[str]
) -> tuple[str, ...]:
    """Declared nonterminals with a production whose rhs mentions the frontier."""
    productions = adapter.production_alternatives
    if isinstance(productions, UnsupportedCapabilityV1):
        return ()
    names = {
        str(production.lhs)
        for production in productions
        for symbol in production.rhs
        if symbol.kind == "terminal" and str(symbol.name) in frontier
    }
    return tuple(sorted(names))


def _meaningful(frontier: frozenset[str]) -> frozenset[str]:
    return frontier - _FRONTIER_IGNORABLE


# --------------------------------------------------------------------------- #
# Certificate: bounded accepting completion through the live frontier
# --------------------------------------------------------------------------- #


def _certify(
    adapter: GrammarCapabilityAdapterV1,
    source: str,
    *,
    starts: tuple[str, ...],
    max_completion_edits: int,
) -> tuple[CompletionCertificateV1, tuple[int, ...]] | None:
    """Prove one bounded accepting continuation, or return None.

    A witness certifies the candidate when the candidate is a string prefix of
    the witness ending at a lexer boundary, the suffix lexes to at most
    ``max_completion_edits`` tokens, every replay step keeps a non-empty
    frontier, and the full witness passes ``static_validate`` (verified
    terminal). The minimum-cost witness wins; ties break by witness source.
    """
    authority = adapter.authority
    witnesses_fn = getattr(authority, "witness_candidates", None)
    if witnesses_fn is None:
        return None
    grammar_path = _grammar_path(adapter)
    if grammar_path is None:
        return None
    best: tuple[CompletionCertificateV1, tuple[int, ...]] | None = None
    for item in sorted(
        witnesses_fn(), key=lambda candidate: str(candidate.source)
    ):
        witness = str(item.source)
        if not witness.startswith(source) or witness == source:
            continue
        suffix = witness[len(source) :]
        if (
            suffix
            and source[-1] not in _BOUNDARY_CHARS
            and suffix[0] not in _BOUNDARY_CHARS
        ):
            continue  # candidate cut landed mid-token inside this witness
        try:
            tokens = _lex_suffix(grammar_path, starts, suffix)
        except Exception:
            continue
        cost = len(tokens)
        if cost == 0 or cost > max_completion_edits:
            continue
        # Replay: every boundary step must keep a live frontier; record the
        # exact singleton positions (offsets where the frontier pins one
        # content terminal).
        offsets = sorted(
            {
                len(source) + int(getattr(token, "end_pos", 0) or 0)
                for token in tokens
            }
            | {len(witness)}
        )
        singletons: list[int] = []
        replay_ok = True
        for offset in offsets:
            step_frontier = _frontier(adapter, witness[:offset])
            if step_frontier is None or not step_frontier:
                replay_ok = False
                break
            if len(_meaningful(step_frontier)) == 1:
                singletons.append(offset)
        if not replay_ok:
            continue
        try:
            adapter.static_validate(witness)
        except Exception:
            continue
        certificate = CompletionCertificateV1(
            witness_source=witness,
            completion_suffix=suffix,
            completion_cost=cost,
            replay_steps=len(offsets),
            terminal_verified=True,
        )
        if best is None or cost < best[0].completion_cost:
            best = (certificate, tuple(singletons))
    return best


# --------------------------------------------------------------------------- #
# Classifier
# --------------------------------------------------------------------------- #


def _symbols(source: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    runtime = runtime_symbols_for_payload(source)
    roles = tuple(sorted({symbol.role for symbol in runtime}))
    binders = tuple(
        sorted(
            symbol.surface
            for symbol in runtime
            if symbol.role in {"alpha_binder", "fresh_binder"}
        )
    )
    return roles, binders


def classify_partial_state(
    source: str,
    *,
    adapter: GrammarCapabilityAdapterV1,
    max_completion_edits: int = 32,
) -> PartialStateReportV1:
    """Classify one candidate partial source (deterministic, fail-closed)."""
    roles, binders = _symbols(source)

    def report(
        state_class: PartialStateClassV1,
        reason: str,
        **kwargs: Any,
    ) -> PartialStateReportV1:
        return PartialStateReportV1(
            state_class=state_class,
            reason=reason,
            required_roles=roles,
            required_binders=binders,
            **kwargs,
        )

    if not source.strip():
        return report(PartialStateClassV1.INVALID, "empty_source")
    violation = _cut_violation(
        source, placeholder_re=adapter.pack.placeholder_policy.placeholder_re
    )
    if violation is not None:
        return report(
            PartialStateClassV1.INVALID, violation, support_coverage="unknown"
        )
    starts = adapter.start_symbols
    if isinstance(starts, UnsupportedCapabilityV1) or not starts:
        return report(
            PartialStateClassV1.INVALID,
            "capability_unsupported:start_symbols",
            support_coverage="unknown",
        )

    # COMPLETE: parses under the primary document start and validates.
    primary = starts[0]
    parsed_primary = False
    try:
        adapter.fragment_parse(primary, source)
        adapter.static_validate(source)
        parsed_primary = True
    except Exception:
        parsed_primary = False
    if parsed_primary:
        return report(
            PartialStateClassV1.COMPLETE,
            "verified_document",
            start_symbol=primary,
            support_coverage="full",
            min_completion_cost=0,
        )

    # FRAGMENT: parses under another named, declared start symbol.
    for start in sorted(starts):
        if start == primary:
            continue
        try:
            adapter.fragment_parse(start, source)
        except Exception:
            continue
        return report(
            PartialStateClassV1.FRAGMENT,
            "named_fragment",
            start_symbol=start,
            support_coverage="full",
        )

    # PREFIX: live frontier plus a compiler certificate.
    frontier = _frontier(adapter, source)
    if frontier is None:
        return report(
            PartialStateClassV1.INVALID,
            "capability_unsupported:completion_frontier",
            support_coverage="unknown",
        )
    if not frontier:
        return report(
            PartialStateClassV1.INVALID,
            "unreachable_from_compiler",
            support_coverage="unknown",
        )
    certified = _certify(
        adapter,
        source,
        starts=tuple(str(start) for start in starts),
        max_completion_edits=max_completion_edits,
    )
    if certified is None:
        return report(
            PartialStateClassV1.INVALID,
            "no_certified_completion",
            frontier=tuple(sorted(frontier)),
            open_nonterminals=_open_nonterminals(adapter, frontier),
            support_coverage="partial",
        )
    certificate, singletons = certified
    return report(
        PartialStateClassV1.PREFIX,
        "certified_accepting_continuation",
        frontier=tuple(sorted(frontier)),
        open_nonterminals=_open_nonterminals(adapter, frontier),
        exact_singleton_positions=singletons,
        support_coverage="full",
        min_completion_cost=certificate.completion_cost,
        certificate=certificate,
    )


# --------------------------------------------------------------------------- #
# Guards: coverage never authorizes destructive pruning; family stop rule
# --------------------------------------------------------------------------- #


def authorize_pruning(report: PartialStateReportV1) -> PartialStateReportV1:
    """Return the report when pruning is authorized; raise otherwise.

    ``UNKNOWN`` or ``PARTIAL`` support coverage never authorizes destructive
    pruning, and an admitted ``PREFIX`` must carry its compiler certificate.
    """
    if report.support_coverage != "full":
        raise PartialStatePruneError(
            f"coverage {report.support_coverage!r} never authorizes pruning "
            f"({report.state_class.value}:{report.reason})"
        )
    if report.state_class is PartialStateClassV1.INVALID:
        raise PartialStatePruneError(f"invalid state: {report.reason}")
    if (
        report.state_class is PartialStateClassV1.PREFIX
        and report.certificate is None
    ):
        raise PartialStatePruneError("PREFIX without a compiler certificate")
    return report


_UNREACHABLE_REASONS = frozenset(
    {"unreachable_from_compiler", "capability_unsupported:completion_frontier"}
)


def family_valid(reports: Iterable[PartialStateReportV1]) -> bool:
    """Stop rule: any partial unreachable from the live compiler/decoder
    invalidates its family."""
    return not any(
        report.state_class is PartialStateClassV1.INVALID
        and report.reason in _UNREACHABLE_REASONS
        for report in reports
    )


__all__ = [
    "CLASSIFIER_VERSION",
    "SCHEMA_VERSION",
    "CompletionCertificateV1",
    "PartialStateClassV1",
    "PartialStatePruneError",
    "PartialStateReportV1",
    "SupportCoverageV1",
    "authorize_pruning",
    "classify_partial_state",
    "family_valid",
]
