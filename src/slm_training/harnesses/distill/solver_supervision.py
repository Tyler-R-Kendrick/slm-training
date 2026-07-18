"""VSS3-01 (SLM-69): replay-verified solver supervision corpus builder.

Turns SLM-64 solver traces into a versioned training corpus whose targets are
*candidate support sets* and *exact search cost-to-go* — not one arbitrary
reference program. Only replay-valid solver states and certificates produce hard
labels; everything uncertain is ``UNKNOWN`` (neither positive nor negative) or a
censored cost.

Torch-free. Runs no model, trains nothing, makes no quality/ship claim. Fixture
rows built with this module establish wiring only, never model quality.

Row kinds (discriminated by ``row_kind``):

* ``support_set`` — one row per exact-closure solver state and hole, listing the
  domain partitioned into ``supported`` / ``unsupported`` / ``unknown`` with the
  replayable certificate and witness digests behind each hard verdict. **All**
  supported alternatives are retained; the trajectory's chosen value is not
  privileged.
* ``candidate_cost`` — one row per live candidate a decision considered, carrying
  the observed search cost-to-go over the trajectory suffix. Cost is marked
  observed only when the suffix is fully present and replayable; truncated or
  budget-stopped suffixes are censored (``cost_observed=False``).

Honesty invariants enforced here (see ``docs/design/verified-scope-solver.md``
and ``docs/design/published-training-corpus.md``):

1. A trace is replayed (``replay_violations``) before any row is emitted; a trace
   with replay violations produces no hard labels.
2. ``UNSUPPORTED`` is emitted only from a ``certified_deduction`` whose
   certificate replays (guaranteed by (1) in ``full`` certificate mode).
3. ``SUPPORTED`` is emitted only from a verifier-accepted witness carrying a
   certificate/witness digest.
4. ``UNKNOWN`` covers partial coverage / missing capability / budget / stale
   evidence / truncation; it is never converted to a negative.
5. Support-set targets keep every supported alternative.
6. Cost-to-go is observed only for a fully present, replayable suffix; timeouts
   and truncated suffixes are censored.
7. A local ``nogood`` is a hard-negative *feature*, never a global
   ``UNSUPPORTED`` relabel.
8. Rows come from actual recorded on-policy solver states.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from slm_training.dsl.solver.replay import SOLVER_EVENT_KINDS, solver_trace_counters
from slm_training.harnesses.distill.trace_store import replay_violations

SUPERVISION_SCHEMA_VERSION = 1

# Splits treated as held-out for the train-vs-heldout leakage guard.
_HELD_OUT_SPLITS = frozenset({"val", "validation", "dev", "test", "heldout", "held_out"})


# --------------------------------------------------------------------------- #
# Canonicalization (must match dsl.solver.replay so value keys line up)
# --------------------------------------------------------------------------- #


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _digest(obj: Any) -> str:
    return hashlib.sha256(_canonical(obj).encode()).hexdigest()


def _hole_key(hole_dict: dict) -> str:
    return _canonical(hole_dict)


def _value_key(value_dict: dict) -> str:
    return _canonical(value_dict)


# --------------------------------------------------------------------------- #
# Row schema
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SupportSetRow:
    """One exact-closure state × hole: the domain partitioned by support verdict."""

    problem_id: str
    state_fingerprint: str
    parent_fingerprint: str | None
    pack_id: str | None
    constraint_version: str | None
    bounds: dict[str, int]
    oracle_backend_version: str | None
    program_family_id: str
    lineage_id: str
    split_group_id: str
    split: str
    checkpoint_sha: str | None
    decode_config_hash: str | None
    seed: int | None
    capsule_id: str | None
    hole_id: dict[str, Any]
    hole_kind: str | None
    decision_level: int
    domain_values: list[dict[str, Any]]
    supported_values: list[dict[str, Any]]
    unsupported_values: list[dict[str, Any]]
    unknown_values: list[dict[str, Any]]
    certificate_ids_by_value: dict[str, list[str]]
    witness_digests_by_value: dict[str, str]
    state_summary: dict[str, Any]
    final_trajectory_status: str | None
    row_kind: str = "support_set"
    schema_version: int = SUPERVISION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateCostRow:
    """One live candidate a decision considered, with observed search cost-to-go."""

    problem_id: str
    state_fingerprint: str
    program_family_id: str
    lineage_id: str
    split_group_id: str
    split: str
    checkpoint_sha: str | None
    decode_config_hash: str | None
    seed: int | None
    capsule_id: str | None
    hole_id: dict[str, Any]
    hole_kind: str | None
    candidate: dict[str, Any]
    ranker_id: str | None
    chosen: bool
    support_verdict: str
    remaining_expanded_nodes: int
    remaining_verifier_calls: int
    remaining_backtracks: int
    remaining_decisions: int
    terminal_success: bool
    cost_observed: bool
    censor_reason: str | None
    conflict_reason_codes: list[str]
    row_kind: str = "candidate_cost"
    schema_version: int = SUPERVISION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Trace classification + lineage
# --------------------------------------------------------------------------- #


def is_solver_trace(trace: dict) -> bool:
    """True iff the trace carries at least one solver-transition event."""
    return any(
        e.get("kind") in SOLVER_EVENT_KINDS for e in trace.get("events", []) or []
    )


def _lineage(trace: dict, problem_id: str) -> tuple[dict[str, str], bool]:
    """Resolve split/lineage identity, inheriting from the root ProgramSpec.

    Reads ``program_family_id``/``lineage_id``/``split_group_id``/``split`` from
    the trace meta (where the recorder copies them from the root ``ProgramSpec``).
    When absent they are *derived* from ``problem_id`` with ``split='train'`` and
    the trace is flagged so the validation report records that its split lineage
    was not authoritative. The cross-split fingerprint guard is the hard
    anti-leakage enforcement regardless.
    """
    meta = trace.get("meta") or {}
    derived = False
    family = meta.get("program_family_id") or trace.get("program_family_id")
    lineage = meta.get("lineage_id") or trace.get("lineage_id")
    group = meta.get("split_group_id") or trace.get("split_group_id")
    split = meta.get("split") or trace.get("split")
    if not (family and lineage and group and split):
        derived = True
        family = family or problem_id
        lineage = lineage or problem_id
        group = group or problem_id
        split = split or "train"
    return (
        {
            "program_family_id": str(family),
            "lineage_id": str(lineage),
            "split_group_id": str(group),
            "split": str(split),
        },
        derived,
    )


# --------------------------------------------------------------------------- #
# Build result
# --------------------------------------------------------------------------- #


@dataclass
class BuildResult:
    """Outcome of a build: rows, rejections, balancing stats, provenance."""

    support_rows: list[SupportSetRow] = field(default_factory=list)
    cost_rows: list[CandidateCostRow] = field(default_factory=list)
    rejected_traces: list[dict[str, Any]] = field(default_factory=list)
    rejected_rows: list[dict[str, Any]] = field(default_factory=list)
    derived_lineage_traces: list[str] = field(default_factory=list)
    source_trace_ids: list[str] = field(default_factory=list)
    balancing: dict[str, Any] = field(default_factory=dict)

    def counts(self) -> dict[str, int]:
        return {
            "support_set_rows": len(self.support_rows),
            "candidate_cost_rows": len(self.cost_rows),
            "rejected_traces": len(self.rejected_traces),
            "rejected_rows": len(self.rejected_rows),
            "source_traces": len(self.source_trace_ids),
        }


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #


class SolverSupervisionBuilder:
    """Replay-verified builder from solver traces to a support/cost corpus."""

    def __init__(
        self,
        *,
        verify_replay: bool = True,
        oracle_backend_version: str | None = None,
    ) -> None:
        self.verify_replay = bool(verify_replay)
        self.oracle_backend_version = oracle_backend_version

    def build(self, traces: Iterable[dict]) -> BuildResult:
        result = BuildResult()
        # Fingerprint -> split, to reject the same state landing in two splits.
        fp_split: dict[str, str] = {}
        seen_row_digests: set[str] = set()

        for trace in traces:
            trace_id = str(
                trace.get("trajectory_id")
                or trace.get("trace_id")
                or f"idx{len(result.source_trace_ids)}"
            )
            if not is_solver_trace(trace):
                result.rejected_traces.append(
                    {"trace_id": trace_id, "reason": "non_solver_trace"}
                )
                continue
            if self.verify_replay:
                violations = replay_violations(trace)
                if violations:
                    result.rejected_traces.append(
                        {
                            "trace_id": trace_id,
                            "reason": "replay_violations",
                            "violations": violations[:8],
                        }
                    )
                    continue
            result.source_trace_ids.append(trace_id)
            self._emit_trace(
                trace, trace_id, result, fp_split, seen_row_digests
            )

        result.balancing = self._balancing(result)
        return result

    # -- per-trace ---------------------------------------------------------- #

    def _emit_trace(
        self,
        trace: dict,
        trace_id: str,
        result: BuildResult,
        fp_split: dict[str, str],
        seen_row_digests: set[str],
    ) -> None:
        events = [
            e for e in (trace.get("events") or []) if e.get("kind") in SOLVER_EVENT_KINDS
        ]
        meta = trace.get("meta") or {}
        states = [e for e in events if e.get("kind") == "solver_state"]
        if not states:
            result.rejected_traces.append(
                {"trace_id": trace_id, "reason": "no_solver_state"}
            )
            return
        root = states[0]
        problem_id = str(root.get("problem_id") or meta.get("problem_id") or trace_id)
        lineage, derived = _lineage(trace, problem_id)
        if derived:
            result.derived_lineage_traces.append(trace_id)
        split = lineage["split"]

        checkpoint_sha = meta.get("checkpoint_sha")
        decode_config_hash = meta.get("decode_config_hash")
        seed = meta.get("seed")
        oracle_version = (
            self.oracle_backend_version
            or meta.get("oracle_backend_version")
            or (trace.get("solver") or {}).get("oracle_backend_version")
        )

        terminal = next(
            (e for e in reversed(events) if e.get("kind") == "solver_terminal"), None
        )
        status = terminal.get("status") if terminal else None
        truncated = bool(terminal.get("trace_truncated")) if terminal else False
        truncated = truncated or any(e.get("trace_truncated") for e in events)

        recorded_fps = {e.get("state_fingerprint") for e in states}
        # parent map from destructive transitions (after -> before).
        parents: dict[str, str] = {}
        for e in events:
            after = e.get("after_fingerprint")
            before = e.get("before_fingerprint")
            if after and before and after not in parents:
                parents[after] = before

        # Aggregate support evidence per (state_fingerprint, hole_key).
        support: dict[tuple[str, str], dict[str, Any]] = defaultdict(
            lambda: {
                "supported": [],
                "unsupported": [],
                "unknown": [],
                "cert_by_value": defaultdict(list),
                "witness_by_value": {},
                "verdict_by_value": {},
            }
        )

        def _target_fp(fp: str | None) -> str:
            # Attach evidence on unrecorded intermediate fingerprints (closure
            # chains) to the root exact-closure state they derive from.
            return fp if fp in recorded_fps else root.get("state_fingerprint")

        for e in events:
            kind = e.get("kind")
            if kind == "support_result":
                fp = _target_fp(e.get("state_fingerprint"))
                hole = _hole_key(e.get("hole_id", {}))
                cand = e.get("candidate", {})
                vk = _value_key(cand)
                verdict = e.get("verdict")
                bucket = support[(fp, hole)]
                if verdict == "supported":
                    bucket["supported"].append(cand)
                    if e.get("certificate_id"):
                        bucket["cert_by_value"][vk].append(e["certificate_id"])
                    if e.get("witness_digest"):
                        bucket["witness_by_value"][vk] = e["witness_digest"]
                    bucket["verdict_by_value"][vk] = "supported"
                elif verdict == "unknown":
                    bucket["unknown"].append(cand)
                    bucket["verdict_by_value"].setdefault(vk, "unknown")
                elif verdict == "unsupported":
                    bucket["unsupported"].append(cand)
                    if e.get("certificate_id"):
                        bucket["cert_by_value"][vk].append(e["certificate_id"])
                    bucket["verdict_by_value"][vk] = "unsupported"
            elif kind == "certified_deduction":
                fp = _target_fp(e.get("before_fingerprint"))
                hole = _hole_key(e.get("hole_id", {}))
                bucket = support[(fp, hole)]
                certs = list(e.get("certificate_ids", []))
                for cand in e.get("removed", []):
                    vk = _value_key(cand)
                    bucket["unsupported"].append(cand)
                    if certs:
                        bucket["cert_by_value"][vk].extend(certs)
                    bucket["verdict_by_value"][vk] = "unsupported"

        # -- support_set rows: one per recorded state × hole ---------------- #
        for state in states:
            fp = state.get("state_fingerprint")
            if not self._pin_split(fp, split, trace_id, result, fp_split):
                continue
            domain = state.get("domain") or {}
            level = int(state.get("decision_level", 0))
            summary = state.get("domain_summary") or {}
            for hole_key, value_keys in domain.items():
                hole_id = json.loads(hole_key)
                bucket = support.get((fp, hole_key))
                domain_values = [json.loads(vk) for vk in value_keys]
                supported = _dedup_values(bucket["supported"]) if bucket else []
                unsupported = _dedup_values(bucket["unsupported"]) if bucket else []
                unknown = _dedup_values(bucket["unknown"]) if bucket else []
                cert_by_value = (
                    {k: sorted(set(v)) for k, v in bucket["cert_by_value"].items()}
                    if bucket
                    else {}
                )
                witness_by_value = dict(bucket["witness_by_value"]) if bucket else {}
                row = SupportSetRow(
                    problem_id=problem_id,
                    state_fingerprint=fp,
                    parent_fingerprint=parents.get(fp),
                    pack_id=state.get("pack_id"),
                    constraint_version=state.get("constraint_version"),
                    bounds=state.get("bounds") or {},
                    oracle_backend_version=oracle_version,
                    program_family_id=lineage["program_family_id"],
                    lineage_id=lineage["lineage_id"],
                    split_group_id=lineage["split_group_id"],
                    split=split,
                    checkpoint_sha=checkpoint_sha,
                    decode_config_hash=decode_config_hash,
                    seed=seed,
                    capsule_id=state.get("capsule_id") or hole_id.get("capsule_id"),
                    hole_id=hole_id,
                    hole_kind=hole_id.get("kind"),
                    decision_level=level,
                    domain_values=domain_values,
                    supported_values=supported,
                    unsupported_values=unsupported,
                    unknown_values=unknown,
                    certificate_ids_by_value=cert_by_value,
                    witness_digests_by_value=witness_by_value,
                    state_summary=summary,
                    final_trajectory_status=status,
                )
                self._add_row(row, result.support_rows, seen_row_digests, result)

        # -- candidate_cost rows: one per live candidate per decision ------- #
        decisions = [
            (i, e) for i, e in enumerate(events) if e.get("kind") == "decision"
        ]
        for i, decision in decisions:
            before = decision.get("before_fingerprint")
            if not self._pin_split(before, split, trace_id, result, fp_split):
                continue
            hole_id = decision.get("hole_id", {})
            hole_key = _hole_key(hole_id)
            chosen = decision.get("chosen", {})
            alternatives = decision.get("alternatives", [])
            ranker_id = decision.get("ranker_id")
            verdict_by_value = (
                support.get((_target_fp(before), hole_key), {}).get(
                    "verdict_by_value", {}
                )
            )
            # Observed cost-to-go over the suffix after this decision.
            suffix_events = events[i + 1 :]
            counts = solver_trace_counters(suffix_events)
            suffix_truncated = truncated or any(
                e.get("trace_truncated") for e in suffix_events
            )
            budget_stop = status == "budget_exhausted" or any(
                str(e.get("stop_reason", "")).startswith("budget")
                for e in suffix_events
            )
            cost_observed = (
                status in {"solved", "certified_unsat"}
                and not suffix_truncated
                and not budget_stop
            )
            censor_reason = None
            if not cost_observed:
                if suffix_truncated:
                    censor_reason = "truncated_suffix"
                elif budget_stop:
                    censor_reason = "budget_exhausted"
                else:
                    censor_reason = "nonterminal_status"
            conflict_codes = _conflict_codes(suffix_events)
            terminal_success = status == "solved"

            for cand, is_chosen in [(chosen, True)] + [
                (alt, False) for alt in alternatives
            ]:
                vk = _value_key(cand)
                row = CandidateCostRow(
                    problem_id=problem_id,
                    state_fingerprint=before,
                    program_family_id=lineage["program_family_id"],
                    lineage_id=lineage["lineage_id"],
                    split_group_id=lineage["split_group_id"],
                    split=split,
                    checkpoint_sha=checkpoint_sha,
                    decode_config_hash=decode_config_hash,
                    seed=seed,
                    capsule_id=hole_id.get("capsule_id"),
                    hole_id=hole_id,
                    hole_kind=hole_id.get("kind"),
                    candidate=cand,
                    ranker_id=ranker_id,
                    chosen=is_chosen,
                    support_verdict=verdict_by_value.get(vk, "unobserved"),
                    remaining_expanded_nodes=counts["solver_states"]
                    + counts["decisions"],
                    remaining_verifier_calls=counts["support_supported"]
                    + counts["support_unknown"]
                    + counts["support_unsupported"],
                    remaining_backtracks=counts["backtracks"],
                    remaining_decisions=counts["decisions"],
                    terminal_success=terminal_success,
                    cost_observed=cost_observed,
                    censor_reason=censor_reason,
                    conflict_reason_codes=conflict_codes,
                )
                self._add_row(row, result.cost_rows, seen_row_digests, result)

    # -- helpers ------------------------------------------------------------ #

    def _pin_split(
        self,
        fp: str | None,
        split: str,
        trace_id: str,
        result: BuildResult,
        fp_split: dict[str, str],
    ) -> bool:
        """Reject a state fingerprint that already lives in a different split."""
        if not fp:
            return True
        prior = fp_split.get(fp)
        if prior is None:
            fp_split[fp] = split
            return True
        if prior == split:
            return True
        # Cross-split appearance: a leak between train and held-out.
        if {prior, split} & _HELD_OUT_SPLITS:
            result.rejected_rows.append(
                {
                    "trace_id": trace_id,
                    "reason": "cross_split_state_leak",
                    "state_fingerprint": fp,
                    "splits": sorted({prior, split}),
                }
            )
            return False
        return True

    def _add_row(
        self,
        row: SupportSetRow | CandidateCostRow,
        sink: list,
        seen_row_digests: set[str],
        result: BuildResult,
    ) -> None:
        digest = _digest(row.to_dict())
        if digest in seen_row_digests:
            return
        seen_row_digests.add(digest)
        sink.append(row)

    def _balancing(self, result: BuildResult) -> dict[str, Any]:
        """Transparent derived statistics — computed, never silently applied."""
        verdict_freq: Counter[str] = Counter()
        domain_sizes: Counter[int] = Counter()
        capsule_widths: Counter[Any] = Counter()
        state_repeats: Counter[str] = Counter()
        for row in result.support_rows:
            verdict_freq["supported"] += len(row.supported_values)
            verdict_freq["unsupported"] += len(row.unsupported_values)
            verdict_freq["unknown"] += len(row.unknown_values)
            domain_sizes[len(row.domain_values)] += 1
            capsule_widths[row.capsule_id] += 1
            state_repeats[row.state_fingerprint] += 1
        observed_costs = sorted(
            r.remaining_decisions for r in result.cost_rows if r.cost_observed
        )
        total_verdicts = sum(verdict_freq.values()) or 1
        n_classes = max(1, len([v for v in verdict_freq.values() if v]))
        inverse_freq_weights = {
            verdict: round(total_verdicts / (n_classes * count), 6)
            for verdict, count in verdict_freq.items()
            if count
        }
        return {
            "formula_version": SUPERVISION_SCHEMA_VERSION,
            "support_verdict_frequencies": dict(verdict_freq),
            "domain_size_distribution": {
                str(k): v for k, v in sorted(domain_sizes.items())
            },
            "capsule_interface_width_distribution": {
                str(k): v for k, v in capsule_widths.items()
            },
            "cost_quantiles_remaining_decisions": _quantiles(observed_costs),
            "repeated_state_frequency": {
                fp: n for fp, n in state_repeats.items() if n > 1
            },
            "suggested_inverse_frequency_weights": inverse_freq_weights,
            "weight_formula": "total / (n_nonempty_classes * class_count)",
            "cost_observed_rows": len(observed_costs),
            "cost_censored_rows": sum(
                1 for r in result.cost_rows if not r.cost_observed
            ),
        }


def _dedup_values(values: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for value in values:
        vk = _value_key(value)
        if vk not in seen:
            seen.add(vk)
            out.append(value)
    return out


def _conflict_codes(events: list[dict]) -> list[str]:
    codes: list[str] = []
    for e in events:
        if e.get("kind") == "backtrack" and e.get("conflict_kind"):
            codes.append(f"backtrack:{e['conflict_kind']}")
        elif e.get("kind") == "nogood":
            provenance = e.get("provenance")
            if isinstance(provenance, dict) and provenance.get("kind"):
                codes.append(f"nogood:{provenance['kind']}")
            elif provenance:
                codes.append("nogood")
    # Deterministic, de-duplicated.
    return sorted(set(codes))


def _quantiles(values: list[int]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)

    def q(p: float) -> float:
        if len(ordered) == 1:
            return float(ordered[0])
        idx = p * (len(ordered) - 1)
        low = int(idx)
        high = min(low + 1, len(ordered) - 1)
        frac = idx - low
        return round(ordered[low] * (1 - frac) + ordered[high] * frac, 4)

    return {"p10": q(0.1), "p50": q(0.5), "p90": q(0.9), "max": float(ordered[-1])}


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def _sorted_rows(rows: list) -> list[dict]:
    payloads = [r.to_dict() for r in rows]
    payloads.sort(key=_canonical)
    return payloads


def write_corpus(
    result: BuildResult,
    output_dir: Path | str,
    *,
    build_command: list[str] | str | None = None,
) -> dict[str, Any]:
    """Write append-only rows by kind/split + manifest + validation report."""
    out = Path(output_dir)
    rows_dir = out / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)

    partitions: dict[str, list[dict]] = defaultdict(list)
    for payload in _sorted_rows(result.support_rows):
        partitions[f"support_set.{payload['split']}"].append(payload)
    for payload in _sorted_rows(result.cost_rows):
        partitions[f"candidate_cost.{payload['split']}"].append(payload)

    artifacts: dict[str, dict[str, Any]] = {}
    for name, payloads in sorted(partitions.items()):
        path = rows_dir / f"{name}.jsonl"
        # Append-only: never rewrite existing rows.
        with path.open("a", encoding="utf-8") as handle:
            for payload in payloads:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
        artifacts[name] = {
            "path": path.relative_to(out).as_posix(),
            "rows": len(payloads),
            "content_sha256": _sha256_file(path),
        }

    manifest = {
        "kind": "solver_supervision",
        "schema_version": SUPERVISION_SCHEMA_VERSION,
        "append_only": True,
        "counts": result.counts(),
        "counts_by_verdict": result.balancing.get(
            "support_verdict_frequencies", {}
        ),
        "counts_by_censor_reason": _censor_histogram(result),
        "counts_by_pack": _pack_histogram(result),
        "counts_by_capsule": result.balancing.get(
            "capsule_interface_width_distribution", {}
        ),
        "counts_by_hole_kind": _hole_kind_histogram(result),
        "source_trace_ids": result.source_trace_ids,
        "artifacts": artifacts,
        "balancing": result.balancing,
        "build_command": build_command,
        "note": (
            "Fixture/build rows establish wiring only and do not establish "
            "model quality. Balancing weights are derived fields, not applied."
        ),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    validation = {
        "schema_version": SUPERVISION_SCHEMA_VERSION,
        "rejected_traces": result.rejected_traces,
        "rejected_rows": result.rejected_rows,
        "derived_lineage_traces": result.derived_lineage_traces,
    }
    (out / "validation_report.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _censor_histogram(result: BuildResult) -> dict[str, int]:
    hist: Counter[str] = Counter()
    for row in result.cost_rows:
        hist[row.censor_reason or "observed"] += 1
    return dict(hist)


def _pack_histogram(result: BuildResult) -> dict[str, int]:
    hist: Counter[str] = Counter()
    for row in result.support_rows:
        hist[str(row.pack_id)] += 1
    return dict(hist)


def _hole_kind_histogram(result: BuildResult) -> dict[str, int]:
    hist: Counter[str] = Counter()
    for row in result.support_rows:
        hist[str(row.hole_kind)] += 1
    return dict(hist)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# High-level entry point
# --------------------------------------------------------------------------- #


def build_solver_supervision(
    traces: Iterable[dict],
    output_dir: Path | str | None = None,
    *,
    verify_replay: bool = True,
    oracle_backend_version: str | None = None,
    dry_run: bool = False,
    build_command: list[str] | str | None = None,
) -> dict[str, Any]:
    """Build (and optionally persist) a solver supervision corpus.

    With ``dry_run=True`` (or ``output_dir=None``) nothing is written; the
    returned summary reports the counts and validation errors a full build would
    produce.
    """
    builder = SolverSupervisionBuilder(
        verify_replay=verify_replay,
        oracle_backend_version=oracle_backend_version,
    )
    result = builder.build(traces)
    summary = {
        "schema_version": SUPERVISION_SCHEMA_VERSION,
        "counts": result.counts(),
        "counts_by_censor_reason": _censor_histogram(result),
        "counts_by_hole_kind": _hole_kind_histogram(result),
        "balancing": result.balancing,
        "rejected_traces": result.rejected_traces,
        "rejected_rows": result.rejected_rows,
        "derived_lineage_traces": result.derived_lineage_traces,
        "dry_run": bool(dry_run or output_dir is None),
    }
    if not dry_run and output_dir is not None:
        summary["manifest"] = write_corpus(
            result, output_dir, build_command=build_command
        )
        summary["output_dir"] = str(Path(output_dir))
    return summary


def iter_solver_traces(store_or_path: Any) -> Iterator[dict]:
    """Yield solver traces from a ``TraceStore`` or a trace-store root path."""
    from slm_training.harnesses.distill.trace_store import TraceStore

    store = (
        store_or_path
        if isinstance(store_or_path, TraceStore)
        else TraceStore(store_or_path)
    )
    for trace in store.iter_traces():
        if is_solver_trace(trace):
            yield trace


__all__ = [
    "SUPERVISION_SCHEMA_VERSION",
    "BuildResult",
    "CandidateCostRow",
    "SolverSupervisionBuilder",
    "SupportSetRow",
    "build_solver_supervision",
    "is_solver_trace",
    "iter_solver_traces",
    "write_corpus",
]
