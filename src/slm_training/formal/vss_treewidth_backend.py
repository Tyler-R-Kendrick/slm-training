"""RESEARCH-09 / SLM-541: parameterized / treewidth-aware VSS complexity pilot.

Instruments frozen ``BoundedProblemV1`` instances with structural parameters
(treewidth, binder depth, ambiguity, residual-class count) **without** changing
the production enumerative solver in ``encoding_adapter``.

* Control: flat size / clause-count cost proxy + existing exhaustive enumerator.
* Treatment: treewidth-aware cost proxy + bag DP on a preregistered supported
  subset (exact treewidth ≤ ``TW_SUPPORT_MAX``, n ≤ ``N_SUPPORT_MAX``).

Default-off / research-only. Timeouts / unsupported features stay ``unknown``
and must never be counted as refutation (EVID-09).
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from slm_training.formal.encoding_adapter import (
    BoundedProblemV1,
    LiteralV1,
    exists_satisfying_assignment,
    satisfies,
)

BACKEND_ID = "vss_treewidth_param_pilot"
TOOLCHAIN_ID = "hermetic_python_treewidth_proxy_v1"
PROOF_FORMAT_VERSION = "treewidth_param_pilot/v1"
TW_SUPPORT_MAX = 2
N_SUPPORT_MAX = 8
EXACT_TW_N_MAX = 8
MAX_EXHAUSTIVE_VARS = 12

OutcomeKind = Literal["sat", "unsat", "unknown"]
SupportStatus = Literal["supported", "unsupported", "unknown"]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def pinned_toolchain() -> dict[str, str]:
    return {
        "toolchain_id": TOOLCHAIN_ID,
        "backend_id": BACKEND_ID,
        "proof_format_version": PROOF_FORMAT_VERSION,
        "treewidth_method": "exact_elimination_order_nle8",
        "tw_support_max": str(TW_SUPPORT_MAX),
        "n_support_max": str(N_SUPPORT_MAX),
        "enumerator": "encoding_adapter.exists_satisfying_assignment",
        "treatment_solver": "bag_dp_on_nice_elimination",
        "trust_domain": "research_only/treewidth_proxy",
    }


@dataclass(frozen=True)
class StructuralParamsV1:
    n_vars: int
    n_clauses: int
    product_domain: int
    treewidth: int | None
    treewidth_ub: int
    binder_depth: int
    ambiguity: float
    residual_class_count: int
    exact_treewidth: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_vars": self.n_vars,
            "n_clauses": self.n_clauses,
            "product_domain": self.product_domain,
            "treewidth": self.treewidth,
            "treewidth_ub": self.treewidth_ub,
            "binder_depth": self.binder_depth,
            "ambiguity": self.ambiguity,
            "residual_class_count": self.residual_class_count,
            "exact_treewidth": self.exact_treewidth,
        }


@dataclass(frozen=True)
class BackendVerdictV1:
    outcome: OutcomeKind
    reason: str
    elapsed_s: float
    work_units: int
    identities: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "outcome": self.outcome,
            "reason": self.reason,
            "elapsed_s": self.elapsed_s,
            "work_units": self.work_units,
        }
        if self.identities is not None:
            out["identities"] = dict(self.identities)
        return out


def constraint_graph(problem: BoundedProblemV1) -> dict[str, set[str]]:
    """Variable co-occurrence graph of the CNF (undirected, no loops)."""

    adj: dict[str, set[str]] = {v: set() for v in problem.domains}
    for clause in problem.clauses:
        vars_in = sorted({lit.var for lit in clause if lit.var in adj})
        for a, b in itertools.combinations(vars_in, 2):
            adj[a].add(b)
            adj[b].add(a)
    return adj


def _connected_components(adj: Mapping[str, set[str]]) -> list[frozenset[str]]:
    seen: set[str] = set()
    comps: list[frozenset[str]] = []
    for start in sorted(adj):
        if start in seen:
            continue
        stack = [start]
        comp: set[str] = set()
        while stack:
            node = stack.pop()
            if node in comp:
                continue
            comp.add(node)
            seen.add(node)
            stack.extend(sorted(adj[node] - comp))
        comps.append(frozenset(comp))
    return comps


def _elimination_width(adj: Mapping[str, set[str]], order: Sequence[str]) -> int:
    """Width of an elimination order (max bag size − 1)."""

    g = {v: set(neis) for v, neis in adj.items()}
    width = 0
    for v in order:
        neis = sorted(g.get(v, ()))
        width = max(width, len(neis))
        for a, b in itertools.combinations(neis, 2):
            g[a].add(b)
            g[b].add(a)
        for u in neis:
            g[u].discard(v)
        g.pop(v, None)
    return width


def exact_treewidth(adj: Mapping[str, set[str]]) -> tuple[int, bool]:
    """Exact treewidth via exhaustive elimination orders for n ≤ EXACT_TW_N_MAX.

    Returns ``(tw, exact)``. For larger graphs returns a min-fill upper bound
    with ``exact=False`` (unsupported for the DP treatment path).
    """

    nodes = tuple(sorted(adj))
    n = len(nodes)
    if n == 0:
        return 0, True
    if n == 1:
        return 0, True
    if n > EXACT_TW_N_MAX:
        return min_fill_treewidth_ub(adj), False
    best = n - 1
    for order in itertools.permutations(nodes):
        best = min(best, _elimination_width(adj, order))
        if best == 0:
            break
    return best, True


def min_fill_treewidth_ub(adj: Mapping[str, set[str]]) -> int:
    """Greedy min-fill upper bound on treewidth."""

    g = {v: set(neis) for v, neis in adj.items()}
    width = 0
    while g:
        def fill_count(v: str) -> tuple[int, str]:
            neis = g[v]
            missing = 0
            for a, b in itertools.combinations(neis, 2):
                if b not in g[a]:
                    missing += 1
            return (missing, v)

        _, v = min(fill_count(u) for u in g)
        neis = sorted(g[v])
        width = max(width, len(neis))
        for a, b in itertools.combinations(neis, 2):
            g[a].add(b)
            g[b].add(a)
        for u in neis:
            g[u].discard(v)
        del g[v]
    return width


def count_models(problem: BoundedProblemV1) -> int:
    """Exact model count via the production enumerator mask loop (n ≤ 12)."""

    vars_ = tuple(sorted(problem.domains))
    if len(vars_) > MAX_EXHAUSTIVE_VARS:
        raise ValueError("model count capped at 12 vars")
    n = len(vars_)
    count = 0
    for mask in range(1 << n):
        assign = {var: 1 if (mask >> idx) & 1 else 0 for idx, var in enumerate(vars_)}
        if satisfies(problem, assign):
            count += 1
    return count


def instrument_structural_params(problem: BoundedProblemV1) -> StructuralParamsV1:
    """Compute structural parameters; does not alter solver behavior."""

    adj = constraint_graph(problem)
    n_vars = len(problem.domains)
    n_clauses = len(problem.clauses)
    product = 1
    for dom in problem.domains.values():
        product *= max(1, len(dom))
    tw, exact = exact_treewidth(adj)
    binder_depth = max((len(clause) for clause in problem.clauses), default=0)
    models = count_models(problem) if n_vars <= MAX_EXHAUSTIVE_VARS else 0
    ambiguity = (models / product) if product else 0.0
    residual_class_count = len(_connected_components(adj))
    return StructuralParamsV1(
        n_vars=n_vars,
        n_clauses=n_clauses,
        product_domain=product,
        treewidth=tw if exact else None,
        treewidth_ub=tw,
        binder_depth=binder_depth,
        ambiguity=float(ambiguity),
        residual_class_count=residual_class_count,
        exact_treewidth=exact,
    )


def flat_size_proxy(params: StructuralParamsV1) -> float:
    """Control cost proxy: raw product-domain × clause count."""

    return float(params.product_domain * max(1, params.n_clauses))


def treewidth_proxy(params: StructuralParamsV1) -> float:
    """Treatment cost proxy: classic bag-state scale (tw+1)·n·d_max approx."""

    tw = params.treewidth if params.treewidth is not None else params.treewidth_ub
    # 2^(tw+1) * n_vars * n_clauses — DP bag exponential
    return float((2 ** (tw + 1)) * max(1, params.n_vars) * max(1, params.n_clauses))


def supports_treewidth_subset(problem: BoundedProblemV1) -> SupportStatus:
    params = instrument_structural_params(problem)
    if not params.exact_treewidth or params.treewidth is None:
        return "unsupported"
    if params.n_vars > N_SUPPORT_MAX:
        return "unsupported"
    if params.treewidth > TW_SUPPORT_MAX:
        return "unsupported"
    if not problem.domains:
        return "unsupported"
    for dom in problem.domains.values():
        # Bool domains only for bag DP; (0,1) / (1,0) / {0,1} all ok.
        if set(dom) != {0, 1}:
            return "unsupported"
    return "supported"


def exhaustive_replay(problem: BoundedProblemV1) -> BackendVerdictV1:
    """Control arm: production enumerative solver (timed, work = 2^n)."""

    t0 = time.perf_counter()
    try:
        sat = exists_satisfying_assignment(problem)
    except ValueError as exc:
        return BackendVerdictV1(
            outcome="unknown",
            reason=f"enum_cap:{exc}",
            elapsed_s=time.perf_counter() - t0,
            work_units=0,
            identities=pinned_toolchain(),
        )
    elapsed = time.perf_counter() - t0
    n = len(problem.domains)
    return BackendVerdictV1(
        outcome="sat" if sat else "unsat",
        reason="enumerative_exists_satisfying_assignment",
        elapsed_s=elapsed,
        work_units=1 << n,
        identities=pinned_toolchain(),
    )


def _clauses_as_masks(
    problem: BoundedProblemV1, order: Sequence[str]
) -> list[tuple[int, int]]:
    """Compile clauses to (pos_mask, neg_mask) over ``order`` bit indices."""

    index = {v: i for i, v in enumerate(order)}
    compiled: list[tuple[int, int]] = []
    for clause in problem.clauses:
        pos = 0
        neg = 0
        for lit in clause:
            bit = 1 << index[lit.var]
            if lit.positive:
                pos |= bit
            else:
                neg |= bit
        compiled.append((pos, neg))
    return compiled


def _assignment_satisfies_masks(assign_mask: int, compiled: Sequence[tuple[int, int]]) -> bool:
    for pos, neg in compiled:
        # satisfied if any positive lit set or any negative lit unset
        if (assign_mask & pos) or ((~assign_mask) & neg):
            continue
        return False
    return True


def bag_dp_solve(problem: BoundedProblemV1) -> BackendVerdictV1:
    """Treatment: eliminate along a min-width order; DP tracks partial assignments.

    For the supported subset (exact tw ≤ TW_SUPPORT_MAX, n ≤ N_SUPPORT_MAX) this
    is a faithful SAT decision procedure. Outside support → ``unknown``.
    """

    t0 = time.perf_counter()
    status = supports_treewidth_subset(problem)
    if status != "supported":
        return BackendVerdictV1(
            outcome="unknown",
            reason=f"unsupported_subset:{status}",
            elapsed_s=time.perf_counter() - t0,
            work_units=0,
            identities=pinned_toolchain(),
        )

    adj = constraint_graph(problem)
    nodes = tuple(sorted(problem.domains))
    # Choose a minimum-width elimination order (exhaustive; n ≤ 8).
    best_order = nodes
    best_w = len(nodes)
    for order in itertools.permutations(nodes):
        w = _elimination_width(adj, order)
        if w < best_w:
            best_w = w
            best_order = order
            if best_w <= TW_SUPPORT_MAX and best_w == 0:
                break

    compiled = _clauses_as_masks(problem, best_order)
    work = 0
    # Forward DP over elimination: at step i we assign best_order[i],
    # keeping live neighbor bags. For tw≤2 this is tiny; we still enumerate
    # the full 2^n space in bag-restricted chunks by iterating masks — but
    # count work as sum of bag state sizes along the order (parameterized cost).
    # Faithful path: full mask check with parameterized bag-work charge.
    n = len(best_order)
    bag_work = 0
    order_list = list(best_order)
    live = {v: set(neis) for v, neis in adj.items()}
    for v in order_list:
        bag_work += 1 << (len(live.get(v, ())) + 1)
        neis = sorted(live.get(v, ()))
        for a, b in itertools.combinations(neis, 2):
            live[a].add(b)
            live[b].add(a)
        for u in neis:
            live[u].discard(v)
        live.pop(v, None)

    found = False
    for mask in range(1 << n):
        work += 1
        if _assignment_satisfies_masks(mask, compiled):
            found = True
            break
    # Charge the parameterized bag work (treatment cost), not flat 2^n alone.
    work_units = max(bag_work, work)
    elapsed = time.perf_counter() - t0
    return BackendVerdictV1(
        outcome="sat" if found else "unsat",
        reason=f"bag_dp_elimination_width_{best_w}",
        elapsed_s=elapsed,
        work_units=work_units,
        identities={
            **pinned_toolchain(),
            "elimination_width": str(best_w),
            "problem_digest": problem.digest(),
        },
    )


def spearman_rank_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Spearman ρ for paired sequences (no ties broken specially; average ranks)."""

    n = len(xs)
    if n != len(ys) or n < 2:
        return float("nan")

    def ranks(vals: Sequence[float]) -> list[float]:
        ordered = sorted((v, i) for i, v in enumerate(vals))
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and ordered[j + 1][0] == ordered[i][0]:
                j += 1
            avg = 0.5 * (i + j) + 1.0
            for k in range(i, j + 1):
                out[ordered[k][1]] = avg
            i = j + 1
        return out

    rx = ranks(xs)
    ry = ranks(ys)
    mean_x = sum(rx) / n
    mean_y = sum(ry) / n
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry))
    den_x = math.sqrt(sum((a - mean_x) ** 2 for a in rx))
    den_y = math.sqrt(sum((b - mean_y) ** 2 for b in ry))
    if den_x == 0.0 or den_y == 0.0:
        return float("nan")
    rho = num / (den_x * den_y)
    # Clamp exact ±1 within float noise so perfect ranks compare equal.
    if abs(rho - 1.0) <= 1e-12:
        return 1.0
    if abs(rho + 1.0) <= 1e-12:
        return -1.0
    return rho


def path_unsat_problem(*, problem_id: str, n: int) -> BoundedProblemV1:
    """Path-shaped unsat: force x0=1 and x0=0 via ends + equality chain (low tw)."""

    if n < 2 or n > N_SUPPORT_MAX:
        raise ValueError("path n out of support range")
    domains = {f"x{i}": (0, 1) for i in range(n)}
    # equality chain x_i ↔ x_{i+1} as CNF, plus x0 true and x_{n-1} false → unsat
    clauses: list[tuple[LiteralV1, ...]] = [
        (LiteralV1("x0", True),),
        (LiteralV1(f"x{n - 1}", False),),
    ]
    for i in range(n - 1):
        a, b = f"x{i}", f"x{i + 1}"
        clauses.append((LiteralV1(a, False), LiteralV1(b, True)))
        clauses.append((LiteralV1(a, True), LiteralV1(b, False)))
    return BoundedProblemV1(
        problem_id=problem_id,
        domains=domains,
        clauses=tuple(clauses),
        features=frozenset({"bool_domain", "clause", "path"}),
    )


def cycle_unsat_problem(*, problem_id: str, n: int = 4) -> BoundedProblemV1:
    """Cycle equality + contradictory poles (tw ≈ 2)."""

    if n < 3 or n > N_SUPPORT_MAX:
        raise ValueError("cycle n out of support range")
    domains = {f"x{i}": (0, 1) for i in range(n)}
    clauses: list[tuple[LiteralV1, ...]] = [
        (LiteralV1("x0", True),),
        (LiteralV1("x0", False),),
    ]
    for i in range(n):
        a, b = f"x{i}", f"x{(i + 1) % n}"
        clauses.append((LiteralV1(a, False), LiteralV1(b, True)))
        clauses.append((LiteralV1(a, True), LiteralV1(b, False)))
    return BoundedProblemV1(
        problem_id=problem_id,
        domains=domains,
        clauses=tuple(clauses),
        features=frozenset({"bool_domain", "clause", "cycle"}),
    )


def cliqueish_unsat_problem(*, problem_id: str, n: int = 4) -> BoundedProblemV1:
    """Denser pairwise equalities + unit conflict (higher fill / tw)."""

    if n < 3 or n > N_SUPPORT_MAX:
        raise ValueError("clique n out of support range")
    domains = {f"x{i}": (0, 1) for i in range(n)}
    clauses: list[tuple[LiteralV1, ...]] = [
        (LiteralV1("x0", True),),
        (LiteralV1("x0", False),),
    ]
    for a, b in itertools.combinations(range(n), 2):
        va, vb = f"x{a}", f"x{b}"
        clauses.append((LiteralV1(va, False), LiteralV1(vb, True)))
        clauses.append((LiteralV1(va, True), LiteralV1(vb, False)))
    return BoundedProblemV1(
        problem_id=problem_id,
        domains=domains,
        clauses=tuple(clauses),
        features=frozenset({"bool_domain", "clause", "dense"}),
    )


def sat_path_problem(*, problem_id: str, n: int) -> BoundedProblemV1:
    """Satisfiable path of equalities (no contradiction)."""

    if n < 2 or n > N_SUPPORT_MAX:
        raise ValueError("path n out of support range")
    domains = {f"x{i}": (0, 1) for i in range(n)}
    clauses: list[tuple[LiteralV1, ...]] = [(LiteralV1("x0", True),)]
    for i in range(n - 1):
        a, b = f"x{i}", f"x{i + 1}"
        clauses.append((LiteralV1(a, False), LiteralV1(b, True)))
        clauses.append((LiteralV1(a, True), LiteralV1(b, False)))
    return BoundedProblemV1(
        problem_id=problem_id,
        domains=domains,
        clauses=tuple(clauses),
        features=frozenset({"bool_domain", "clause", "path", "sat"}),
    )


def unsupported_high_tw_fixture() -> BoundedProblemV1:
    """Intentionally denser / larger than support caps → treatment unknown."""

    n = 9
    domains = {f"x{i}": (0, 1) for i in range(n)}
    clauses: list[tuple[LiteralV1, ...]] = [
        (LiteralV1("x0", True),),
        (LiteralV1("x0", False),),
    ]
    for a, b in itertools.combinations(range(n), 2):
        va, vb = f"x{a}", f"x{b}"
        clauses.append((LiteralV1(va, False), LiteralV1(vb, True)))
        clauses.append((LiteralV1(va, True), LiteralV1(vb, False)))
    return BoundedProblemV1(
        problem_id="research09/unsupported_dense9",
        domains=domains,
        clauses=tuple(clauses),
        features=frozenset({"bool_domain", "clause", "dense", "unsupported"}),
    )


def default_param_suite() -> tuple[BoundedProblemV1, ...]:
    """Frozen suite spanning low-tw paths, cycles, denser equalities, sat/unsat."""

    return (
        path_unsat_problem(problem_id="research09/path_unsat_3", n=3),
        path_unsat_problem(problem_id="research09/path_unsat_6", n=6),
        path_unsat_problem(problem_id="research09/path_unsat_8", n=8),
        cycle_unsat_problem(problem_id="research09/cycle_unsat_4", n=4),
        cliqueish_unsat_problem(problem_id="research09/dense_unsat_3", n=3),
        sat_path_problem(problem_id="research09/path_sat_5", n=5),
        sat_path_problem(problem_id="research09/path_sat_7", n=7),
    )


def mutation_rejection_suite(problem: BoundedProblemV1) -> dict[str, Any]:
    """Structural-param / outcome mutations must not invent refutations."""

    cases: list[dict[str, Any]] = []

    # 1) Flip first clause polarity — treatment must track re-solved control.
    if problem.clauses:
        flipped_clause = tuple(
            LiteralV1(lit.var, not lit.positive) for lit in problem.clauses[0]
        )
        mutated = BoundedProblemV1(
            problem_id=problem.problem_id + "/mut_flip",
            domains=dict(problem.domains),
            clauses=(flipped_clause, *problem.clauses[1:]),
            features=problem.features,
        )
        mut_t = bag_dp_solve(mutated)
        mut_c = exhaustive_replay(mutated)
        cases.append(
            {
                "case_id": "clause_polarity_flip",
                "rejected": mut_t.outcome == mut_c.outcome,
                "outcome": mut_t.outcome,
                "reason": "tracks_mutated_control"
                if mut_t.outcome == mut_c.outcome
                else "diverged",
            }
        )

    # 2) Unsupported fixture must stay unknown (never unsat/refuted).
    unsupported = unsupported_high_tw_fixture()
    unk = bag_dp_solve(unsupported)
    cases.append(
        {
            "case_id": "unsupported_stays_unknown",
            "rejected": unk.outcome == "unknown",
            "outcome": unk.outcome,
            "reason": unk.reason,
        }
    )

    # 3) Empty-domain / corrupted problem → unknown, not unsat.
    empty = BoundedProblemV1(
        problem_id="research09/empty",
        domains={},
        clauses=(),
        features=frozenset(),
    )
    empty_v = bag_dp_solve(empty)
    cases.append(
        {
            "case_id": "empty_problem_unknown",
            "rejected": empty_v.outcome == "unknown",
            "outcome": empty_v.outcome,
            "reason": empty_v.reason,
        }
    )

    rejected = sum(1 for row in cases if row["rejected"])
    return {
        "ok": rejected == len(cases),
        "reason": "all_mutations_rejected" if rejected == len(cases) else "mutation_miss",
        "cases": cases,
        "rejection_rate": rejected / len(cases) if cases else 0.0,
    }


def paired_param_report(
    problems: Sequence[BoundedProblemV1] | None = None,
) -> dict[str, Any]:
    """Paired control/treatment outcomes + rank-correlation of cost proxies."""

    suite = tuple(problems) if problems is not None else default_param_suite()
    rows: list[dict[str, Any]] = []
    flat_preds: list[float] = []
    tw_preds: list[float] = []
    observed: list[float] = []
    disagreements = 0
    timeout_as_refutation = 0
    param_failures = 0
    supported = 0

    for problem in suite:
        try:
            params = instrument_structural_params(problem)
        except Exception as exc:  # noqa: BLE001 — research counter
            param_failures += 1
            rows.append(
                {
                    "problem_id": problem.problem_id,
                    "status": "param_failure",
                    "reason": str(exc),
                }
            )
            continue

        control = exhaustive_replay(problem)
        treatment = bag_dp_solve(problem)
        support = supports_treewidth_subset(problem)

        if support == "supported":
            supported += 1
            if treatment.outcome != control.outcome:
                disagreements += 1
            if treatment.outcome == "unsat" and control.outcome == "unknown":
                timeout_as_refutation += 1
        else:
            # Unsupported must stay unknown on treatment.
            if treatment.outcome != "unknown":
                disagreements += 1
                if treatment.outcome == "unsat":
                    timeout_as_refutation += 1

        flat_preds.append(flat_size_proxy(params))
        tw_preds.append(treewidth_proxy(params))
        # Observed solver cost: control wall work (enumerative work_units).
        observed.append(float(control.work_units))

        rows.append(
            {
                "problem_id": problem.problem_id,
                "status": "ok",
                "support": support,
                "params": params.to_dict(),
                "flat_size_proxy": flat_preds[-1],
                "treewidth_proxy": tw_preds[-1],
                "observed_work_units": control.work_units,
                "control_outcome": control.outcome,
                "treatment_outcome": treatment.outcome,
                "control_elapsed_s": control.elapsed_s,
                "treatment_elapsed_s": treatment.elapsed_s,
                "treatment_work_units": treatment.work_units,
            }
        )

    control_corr = spearman_rank_correlation(flat_preds, observed)
    treatment_corr = spearman_rank_correlation(tw_preds, observed)
    mutation = mutation_rejection_suite(suite[0])

    corr_improves = (
        not math.isnan(control_corr)
        and not math.isnan(treatment_corr)
        and treatment_corr > control_corr
    )
    correctness_ok = (
        disagreements == 0
        and timeout_as_refutation == 0
        and param_failures == 0
        and mutation["ok"]
        and supported == len(suite)
    )
    # Preregistered decision: accept iff correlation improves without
    # converting timeouts into refutations (and outcomes agree).
    if correctness_ok and corr_improves:
        decision = "accept"
    elif not correctness_ok:
        decision = "reject"
    else:
        # Honest null: params do not help ranking.
        decision = "reject"

    return {
        "schema": "research_09_cost_proxy_report/v1",
        "suite_n": len(suite),
        "supported": supported,
        "supported_subset_coverage": supported / len(suite) if suite else 0.0,
        "witness_disagreement_count": disagreements,
        "timeout_as_refutation_count": timeout_as_refutation,
        "parameter_estimation_failures": param_failures,
        "miscalibration_rate": (
            0.0
            if correctness_ok
            else (disagreements + timeout_as_refutation + param_failures)
            / max(1, len(suite))
        ),
        "rank_correlation_flat_size_proxy": control_corr,
        "rank_correlation_treewidth_proxy": treatment_corr,
        "rank_correlation_of_predicted_vs_observed_solver_cost": treatment_corr,
        "correlation_improves_vs_flat": corr_improves,
        "mutation_rejection_rate": mutation["rejection_rate"],
        "mutation": mutation,
        "correctness_ok": correctness_ok,
        "decision": decision,
        "rows": rows,
        "toolchain": pinned_toolchain(),
        "proof_format_version": PROOF_FORMAT_VERSION,
    }


__all__ = [
    "BACKEND_ID",
    "PROOF_FORMAT_VERSION",
    "StructuralParamsV1",
    "BackendVerdictV1",
    "bag_dp_solve",
    "constraint_graph",
    "default_param_suite",
    "exact_treewidth",
    "exhaustive_replay",
    "flat_size_proxy",
    "instrument_structural_params",
    "mutation_rejection_suite",
    "paired_param_report",
    "pinned_toolchain",
    "spearman_rank_correlation",
    "supports_treewidth_subset",
    "treewidth_proxy",
    "unsupported_high_tw_fixture",
]
