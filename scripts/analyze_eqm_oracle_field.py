#!/usr/bin/env python3
"""Exact perfect-training Equilibrium-Matching field on finite datasets.

Numerical analysis card (numpy only, no torch, no training). Equilibrium
Matching (arXiv:2510.02300) trains ``f(x_g) ~ (eps - x) * c(g)`` with
``x_g = g*x + (1-g)*eps``, ``g ~ U(0,1)``, ``eps ~ N(0, I_d)``. Under perfect
training the field is the conditional expectation

    f*(xh) = lam * E[ c(g)/(1-g) * (xh - x) | x_g = xh ]

because ``eps - x = (x_g - x)/(1-g)`` exactly. For a finite dataset the
posterior over ``(i, g)`` is proportional to ``N(xh; g x_i, (1-g)^2 I_d)`` times
the uniform prior, i.e. log-weight ``-|xh - g x_i|^2 / (2(1-g)^2) - d log(1-g)``.
This script evaluates that field in closed form (softmax over ``i``, quadrature
over ``g`` on a log-spaced ``u = 1-g`` grid) and tests five hypotheses about
spurious stationary points, saddle structure, the norm-based adaptive stop,
and the residual field at training points. Claim class: ``analysis`` (an
external-paper re-derivation, not a repository model claim).

Usage::

    PYTHONPATH=src python scripts/analyze_eqm_oracle_field.py --out outputs/eqm

Writes ``<stem>.json`` (machine-readable) and ``<stem>.tables.md`` (the
tables embedded verbatim in ``docs/design/eqm-oracle-field-20260902.md``)
into ``--out``; ``--stem`` defaults to ``eqm-oracle-field-20260902``.
Deterministic (seeded) and bounded to well under 60 s on a 4-CPU box (24 s
measured).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

# Thin (K=2..16) matmuls thrash a threaded BLAS, and the 8 MB temporaries at
# d=4096 are re-mmapped/page-faulted on every call: pin one BLAS thread and
# keep freed blocks on the heap. Both must happen before numpy loads.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
try:  # glibc only; harmless elsewhere
    _libc = ctypes.CDLL("libc.so.6")
    _libc.mallopt(-3, 1 << 30)  # M_MMAP_THRESHOLD
    _libc.mallopt(-1, 1 << 30)  # M_TRIM_THRESHOLD
except (OSError, AttributeError):
    pass

import numpy as np  # noqa: E402

from slm_training.harness_core.versioning import build_version_stamp  # noqa: E402

SEED = 20260902
# The audit this card belongs to is about constrained decoding; the field study
# is an external-paper re-derivation, so it stamps the invariant set it informs.
STAMP_COMPONENTS = ("decode.invariants",)
PAPER_ETA = 0.0017  # Table 2 of the paper, GD step size in latent space
PAPER_STEPS = 250
PAPER_LAMBDA = 4.0
PAPER_TRUNC_A = 0.8
PAPER_GMIN_LATENT = 10.0  # adaptive-stop threshold used for B/2 latents (d=4096)


# --------------------------------------------------------------------------- #
# Oracle field
# --------------------------------------------------------------------------- #
class OracleField:
    """Closed-form perfect-training EqM field for a finite dataset ``X``."""

    def __init__(
        self,
        X: np.ndarray,
        *,
        schedule: str = "trunc",
        trunc_a: float = PAPER_TRUNC_A,
        lam: float = PAPER_LAMBDA,
        u_min: float = 1e-6,
        n_gamma: int = 8192,
        n_log: int | None = None,
        split: float = 0.05,
    ) -> None:
        self.X = np.asarray(X, dtype=np.float64)
        self.n, self.d = self.X.shape
        self.sq = np.einsum("nd,nd->n", self.X, self.X)
        self.lam = float(lam)
        self.schedule = schedule
        # Composite grid in u = 1 - gamma: uniform in gamma on [0, 1-split]
        # (resolves the O(1/sqrt d)-wide posterior peaks, including the
        # gamma=0 edge peak of far-from-data points) plus log-spaced u on
        # [u_min, split] (resolves the u -> 0 peak of near-data points).
        n_log = n_log if n_log is not None else max(n_gamma // 8, 16)
        u_uniform = 1.0 - np.linspace(0.0, 1.0 - split, n_gamma, endpoint=False)
        u_log = np.geomspace(split, u_min, n_log)
        u = np.concatenate([u_uniform, u_log])  # decreasing in u
        u = np.sort(u)[::-1].copy()
        gamma = 1.0 - u
        # trapezoid weights for the uniform prior dgamma = |du|
        w = np.empty_like(u)
        w[1:-1] = 0.5 * (u[:-2] - u[2:])
        w[0] = 0.5 * (u[0] - u[1])
        w[-1] = 0.5 * (u[-2] - u[-1])
        if schedule == "trunc":
            c = np.where(gamma <= trunc_a, 1.0, (1.0 - gamma) / (1.0 - trunc_a))
        elif schedule == "linear":
            c = 1.0 - gamma
        else:
            raise ValueError(f"unknown schedule {schedule!r}")
        self.u = u
        self.gamma = gamma
        self.log_wq = np.log(w)
        self.g = lam * c / u  # c(gamma)/(1-gamma) target multiplier
        self.log_g = np.log(self.g)
        self.d_log_u = self.d * np.log(u)

    def _log_weights(self, Xhat: np.ndarray) -> np.ndarray:
        a = np.einsum("md,md->m", Xhat, Xhat)  # |xh|^2
        b = Xhat @ self.X.T  # m,n : xh . x_i
        u2 = 2.0 * self.u**2
        gam = self.gamma
        # -(a - 2 g b + g^2 |x_i|^2) / (2 u^2) - d log u + log(quad weight)
        quad = (
            a[:, None, None]
            - 2.0 * gam[None, None, :] * b[:, :, None]
            + (gam**2)[None, None, :] * self.sq[None, :, None]
        )
        return -quad / u2[None, None, :] - self.d_log_u[None, None, :] + self.log_wq

    def field(self, Xhat: np.ndarray, chunk: int = 512) -> np.ndarray:
        """Return ``f*(xh)`` for each row of ``Xhat`` (shape ``m,d``)."""
        Xhat = np.atleast_2d(np.asarray(Xhat, dtype=np.float64))
        out = np.empty_like(Xhat)
        for s in range(0, Xhat.shape[0], chunk):
            xh = Xhat[s : s + chunk]
            lw = self._log_weights(xh)  # m,n,G
            mx = lw.max(axis=(1, 2), keepdims=True)
            # clamp: exp(-700) ~ 1e-304 is still a normal double; anything
            # smaller is negligible and would only trigger slow subnormals
            p = np.exp(np.maximum(lw - mx, -700.0))
            Z = p.sum(axis=(1, 2))  # m
            W = np.einsum("mng,g->mn", p, self.g) / Z[:, None]  # E[g 1{i}]
            S = W.sum(axis=1)
            out[s : s + chunk] = xh * S[:, None] - W @ self.X
        return out

    def posterior_gamma_mean(self, xh: np.ndarray) -> float:
        lw = self._log_weights(np.atleast_2d(xh))[0]
        p = np.exp(lw - lw.max())
        return float((p.sum(axis=0) * self.gamma).sum() / p.sum())


def rms(F: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(F * F, axis=1))))


def sig(x: float, n: int = 6) -> float:
    """Round to ``n`` significant digits for byte-stable JSON."""
    if x == 0 or not math.isfinite(x):
        return float(x)
    return float(f"{x:.{n}g}")


# --------------------------------------------------------------------------- #
# Item 1 + 2: antipodal midpoint zero and its classification
# --------------------------------------------------------------------------- #
def antipodal_dataset(
    d: int, rng: np.random.Generator, scale: float | None
) -> np.ndarray:
    x1 = rng.standard_normal(d)
    x1 *= (scale if scale is not None else math.sqrt(d)) / np.linalg.norm(x1)
    return np.stack([x1, -x1])


def item_midpoint(
    d: int, rng: np.random.Generator, *, schedule: str, scale: float | None
) -> dict[str, Any]:
    X = antipodal_dataset(d, rng, scale)
    x1 = X[0]
    norm1 = float(np.linalg.norm(x1))
    orc = OracleField(X, schedule=schedule, n_gamma=16384)
    alphas = np.linspace(-1.5, 1.5, 61)
    F_axis = orc.field(alphas[:, None] * x1[None, :])
    f0 = orc.field(np.zeros((1, d)))[0]
    axis_rms = rms(F_axis)
    # scalar profile s(alpha): f(alpha x1) = s(alpha) x1 (by symmetry, exact)
    s = F_axis @ x1 / norm1**2
    off_axis = float(
        np.max(np.linalg.norm(F_axis - s[:, None] * x1[None, :], axis=1)) / axis_rms
    )
    # slope of s at 0+ (f is odd, so s(a)/a is the two-sided secant)
    slope = {}
    for a in (1e-4, 1e-3, 1e-2):
        sa = float(orc.field((a * x1)[None, :])[0] @ x1 / norm1**2)
        slope[f"{a:g}"] = sig(sa / a)
    # directional second derivatives of E (grad E = f): v . (f(hv) - f(-hv)) / 2h
    h = 1e-3 * norm1
    v_axis = x1 / norm1

    def d2E(v: np.ndarray) -> float:
        fp, fm = orc.field(np.stack([h * v, -h * v]))
        return float(v @ (fp - fm) / (2 * h))

    e_axis = d2E(v_axis)
    e_orth = []
    for _ in range(3):
        v = rng.standard_normal(d)
        v -= (v @ v_axis) * v_axis
        v /= np.linalg.norm(v)
        e_orth.append(d2E(v))
    # tanh model check: E[x|alpha x1] = m(alpha) x1 ; compare m'(0) to slope
    kappa_eff = 1.0 - slope["0.0001"] / (e_orth[0] if e_orth[0] else 1.0)
    gm = orc.posterior_gamma_mean(np.zeros(d))
    return {
        "d": d,
        "norm_x1": sig(norm1),
        "schedule": schedule,
        "f_at_midpoint_norm": sig(float(np.linalg.norm(f0))),
        "rms_f_along_axis": sig(axis_rms),
        "f0_over_rms": sig(float(np.linalg.norm(f0)) / axis_rms),
        "max_off_axis_component_over_rms": sig(off_axis),
        "s_profile": {f"{a:.3f}": sig(float(v)) for a, v in zip(alphas, s)},
        "ds_dalpha_at_0plus": slope,
        "d2E_along_axis": sig(e_axis),
        "d2E_orthogonal": [sig(e) for e in e_orth],
        "classification": (
            "saddle (max along data axis, min orthogonally)"
            if e_axis < 0 and all(e > 0 for e in e_orth)
            else "strict local minimum"
            if e_axis > 0 and all(e > 0 for e in e_orth)
            else "other"
        ),
        "kappa_eff_minus_1_over_Eg": sig(-slope["0.0001"] / e_orth[0]),
        "posterior_gamma_mean_at_midpoint": sig(gm),
        "kappa_eff": sig(kappa_eff),
    }


# --------------------------------------------------------------------------- #
# Item 3: Monte-Carlo GD with the paper's adaptive stop
# --------------------------------------------------------------------------- #
def item_gd_montecarlo(
    d: int,
    rng: np.random.Generator,
    *,
    n_traj: int,
    rhos: tuple[float, ...],
    eta: float,
    steps: int,
    schedule: str,
    scale: float | None,
) -> dict[str, Any]:
    X = antipodal_dataset(d, rng, scale)
    x1 = X[0]
    norm1 = float(np.linalg.norm(x1))
    orc = OracleField(X, schedule=schedule, n_gamma=1024, n_log=128, u_min=1e-3)
    ball = 0.1 * norm1
    Xt = rng.standard_normal((n_traj, d))
    fnorm = np.empty((steps + 1, n_traj))
    dmid = np.empty((steps + 1, n_traj))
    ddata = np.empty((steps + 1, n_traj))
    entered_ball = np.zeros(n_traj, dtype=bool)
    for t in range(steps + 1):
        F = orc.field(Xt)
        fnorm[t] = np.linalg.norm(F, axis=1)
        dmid[t] = np.linalg.norm(Xt, axis=1)
        ddata[t] = np.minimum(
            np.linalg.norm(Xt - x1, axis=1), np.linalg.norm(Xt + x1, axis=1)
        )
        entered_ball |= dmid[t] < ball
        if t < steps:
            Xt = Xt - eta * F
    out: dict[str, Any] = {
        "d": d,
        "norm_x1": sig(norm1),
        "n_traj": n_traj,
        "eta": eta,
        "steps": steps,
        "ball_radius": sig(ball),
        "fnorm_at_init_mean": sig(float(fnorm[0].mean())),
        "fnorm_final_mean": sig(float(fnorm[-1].mean())),
        "min_fnorm_over_trajectories": sig(float(fnorm.min())),
        "final_dist_to_data_mean_over_norm": sig(float(ddata[-1].mean() / norm1)),
        "final_dist_to_midpoint_min_over_norm": sig(float(dmid[-1].min() / norm1)),
        "frac_ever_in_midpoint_ball": sig(float(entered_ball.mean())),
        "by_gmin": [],
    }
    for rho in rhos:
        gmin = rho * PAPER_LAMBDA * norm1
        hit = fnorm < gmin  # steps+1, n
        any_hit = hit.any(axis=0)
        first = np.where(any_hit, hit.argmax(axis=0), steps)
        stop_dmid = dmid[first, np.arange(n_traj)]
        stop_ddata = ddata[first, np.arange(n_traj)]
        term_mid = any_hit & (stop_dmid < ball)
        term_data = any_hit & (stop_ddata < ball)
        out["by_gmin"].append(
            {
                "rho": rho,
                "g_min": sig(gmin),
                "frac_terminated": sig(float(any_hit.mean())),
                "frac_terminated_in_midpoint_ball": sig(float(term_mid.mean())),
                "frac_terminated_near_data": sig(float(term_data.mean())),
                "frac_terminated_elsewhere": sig(
                    float((any_hit & ~term_mid & ~term_data).mean())
                ),
                "mean_stop_step": sig(float(first[any_hit].mean()))
                if any_hit.any()
                else None,
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Item 4: residual field at training points, 16 random points
# --------------------------------------------------------------------------- #
def item_training_points(
    d: int,
    rng: np.random.Generator,
    *,
    schedule: str,
    u_mins: tuple[float, ...],
    probe_radius_rel: float,
) -> dict[str, Any]:
    n = 16
    X = rng.standard_normal((n, d))  # unit-RMS coordinates
    ref = rng.standard_normal((64, d))  # N(0,I) reference inits for RMS(f)
    out: dict[str, Any] = {"d": d, "n_points": n, "by_u_min": []}
    for u_min in u_mins:
        orc = OracleField(X, schedule=schedule, u_min=u_min, n_gamma=2048, n_log=256)
        F_ref = orc.field(ref)
        r = rms(F_ref)
        F_data = orc.field(X)
        norms = np.linalg.norm(F_data, axis=1)
        # small probe: x_i + rho |x_i| xi, xi random unit
        xi = rng.standard_normal((n, d))
        xi /= np.linalg.norm(xi, axis=1, keepdims=True)
        radius = probe_radius_rel * np.linalg.norm(X, axis=1, keepdims=True)
        F_probe = orc.field(X + radius * xi)
        probe_norms = np.linalg.norm(F_probe, axis=1)
        # decompose the probe field: radial coefficient along xi
        radial = np.einsum("nd,nd->n", F_probe, xi) / radius[:, 0]
        out["by_u_min"].append(
            {
                "u_min": u_min,
                "gamma_max": 1.0 - u_min,
                "rms_f_at_N01_inits": sig(r),
                "f_at_training_points_over_rms_mean": sig(float(norms.mean() / r)),
                "f_at_training_points_over_rms_max": sig(float(norms.max() / r)),
                "f_at_training_points_abs_max": sig(float(norms.max())),
                "probe_radius_rel": probe_radius_rel,
                "f_at_probe_over_rms_mean": sig(float(probe_norms.mean() / r)),
                "probe_radial_stiffness_mean": sig(float(radial.mean())),
                "expected_stiffness_lambda_over_1_minus_a": sig(
                    PAPER_LAMBDA / (1 - PAPER_TRUNC_A)
                ),
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Item 5: equilateral 3-point dataset, scan for spurious strict local minima
# --------------------------------------------------------------------------- #
def item_equilateral(
    d: int, rng: np.random.Generator, *, radius: float, schedule: str
) -> dict[str, Any]:
    # orthonormal plane basis (identity in d=2, random in d>2)
    if d == 2:
        B = np.eye(2)
    else:
        Q, _ = np.linalg.qr(rng.standard_normal((d, 2)))
        B = Q.T  # 2,d
    P2 = radius * np.array(
        [[1.0, 0.0], [-0.5, math.sqrt(3) / 2], [-0.5, -math.sqrt(3) / 2]]
    )
    X = P2 @ B  # 3,d
    orc = OracleField(X, schedule=schedule, n_gamma=1024, n_log=128, u_min=1e-3)

    def f2(pts2: np.ndarray) -> np.ndarray:
        """In-plane field coordinates for in-plane points (f stays in-plane)."""
        F = orc.field(pts2 @ B)
        return F @ B.T

    # grid scan of |f| in the plane
    lim = 1.6 * radius
    g = np.linspace(-lim, lim, 121)
    GX, GY = np.meshgrid(g, g, indexing="ij")
    pts = np.stack([GX.ravel(), GY.ravel()], axis=1)
    F = f2(pts)
    fn = np.linalg.norm(F, axis=1).reshape(GX.shape)
    ref_rms = float(np.sqrt(np.mean(fn**2)))
    # local minima of |f| on the grid (strict in 8-neighbourhood)
    interior = fn[1:-1, 1:-1]
    is_min = np.ones_like(interior, dtype=bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            is_min &= (
                interior
                < fn[1 + di : fn.shape[0] - 1 + di, 1 + dj : fn.shape[1] - 1 + dj]
            )
    cand = np.argwhere(is_min) + 1
    stationary: list[dict[str, Any]] = []
    seen: list[np.ndarray] = []
    for ci, cj in cand:
        p = np.array([g[ci], g[cj]])
        # Newton refinement on the in-plane 2x2 Jacobian (finite differences)
        for _ in range(30):
            fp = f2(p[None, :])[0]
            if np.linalg.norm(fp) < 1e-9 * max(ref_rms, 1e-12):
                break
            h = 1e-4 * radius
            J = np.empty((2, 2))
            for k in range(2):
                e = np.zeros(2)
                e[k] = h
                J[:, k] = (f2((p + e)[None, :])[0] - f2((p - e)[None, :])[0]) / (2 * h)
            try:
                step = np.linalg.solve(J, fp)
            except np.linalg.LinAlgError:
                break
            if np.linalg.norm(step) > 0.5 * radius:
                step *= 0.5 * radius / np.linalg.norm(step)
            p = p - step
        if any(np.linalg.norm(p - q) < 1e-3 * radius for q in seen):
            continue
        seen.append(p.copy())
        fp = f2(p[None, :])[0]
        h = 1e-4 * radius
        J = np.empty((2, 2))
        for k in range(2):
            e = np.zeros(2)
            e[k] = h
            J[:, k] = (f2((p + e)[None, :])[0] - f2((p - e)[None, :])[0]) / (2 * h)
        Js = 0.5 * (J + J.T)
        ev = np.sort(np.linalg.eigvalsh(Js))
        # orthogonal-to-plane directional second derivative = S(x) = E[g | x] > 0
        if d > 2:
            xh = p @ B
            v = rng.standard_normal(d)
            v -= B.T @ (B @ v)
            v /= np.linalg.norm(v)
            hh = 1e-3 * radius
            fpv, fmv = orc.field(np.stack([xh + hh * v, xh - hh * v]))
            e_orth = float(v @ (fpv - fmv) / (2 * hh))
        else:
            e_orth = None
        dist_data = float(np.min(np.linalg.norm(P2 - p[None, :], axis=1)))
        is_stationary = bool(np.linalg.norm(fp) < 1e-6 * ref_rms)
        all_pos = bool(ev[0] > 0 and (e_orth is None or e_orth > 0))
        stationary.append(
            {
                "point_plane_coords_over_radius": [
                    sig(float(p[0] / radius)),
                    sig(float(p[1] / radius)),
                ],
                "fnorm_over_rms": sig(float(np.linalg.norm(fp) / ref_rms)),
                "is_stationary": is_stationary,
                "dist_to_nearest_datum_over_radius": sig(dist_data / radius),
                "on_data": bool(dist_data < 1e-3 * radius),
                "inplane_hessian_eigs": [sig(float(ev[0])), sig(float(ev[1]))],
                "orthogonal_d2E": sig(e_orth) if e_orth is not None else None,
                "type": (
                    "not stationary"
                    if not is_stationary
                    else "strict local minimum"
                    if all_pos
                    else "local maximum"
                    if ev[1] < 0 and (e_orth is None or e_orth < 0)
                    else "saddle"
                ),
            }
        )
    spurious = [
        s
        for s in stationary
        if s["is_stationary"]
        and not s["on_data"]
        and s["type"] == "strict local minimum"
    ]
    # also a GD sweep from N(0,I) inits: where do trajectories end?
    n_traj = 128
    Xt = rng.standard_normal((n_traj, d))
    for _ in range(PAPER_STEPS):
        Xt = Xt - PAPER_ETA * orc.field(Xt)
    dd = np.linalg.norm(Xt[:, None, :] - X[None, :, :], axis=2).min(axis=1)
    dc = np.linalg.norm(Xt, axis=1)
    return {
        "d": d,
        "radius": radius,
        "grid": {"points_per_side": len(g), "half_width_over_radius": 1.6},
        "rms_f_on_grid": sig(ref_rms),
        "stationary_points": stationary,
        "n_spurious_strict_minima": len(spurious),
        "gd_endpoints": {
            "n_traj": n_traj,
            "frac_within_0p1R_of_datum": sig(float((dd < 0.1 * radius).mean())),
            "frac_within_0p1R_of_centroid": sig(float((dc < 0.1 * radius).mean())),
            "max_dist_to_datum_over_radius": sig(float(dd.max() / radius)),
        },
    }


# --------------------------------------------------------------------------- #
# Quadrature self-check
# --------------------------------------------------------------------------- #
def quadrature_check(d: int, rng: np.random.Generator, schedule: str) -> dict[str, Any]:
    X = antipodal_dataset(d, rng, None)
    pts = rng.standard_normal((8, d))
    axis = np.array(
        [-1.2, -0.9, -0.5, -0.1, 0.1, 0.5, 0.9, 1.2]
    )  # alpha=0 excluded (f=0)
    pts = np.concatenate([pts, axis[:, None] * X[0][None, :], X[0][None, :] * 1.001])
    coarse = OracleField(
        X, schedule=schedule, n_gamma=1024, n_log=128, u_min=1e-3
    ).field(pts)
    fine = OracleField(
        X, schedule=schedule, n_gamma=32768, n_log=4096, u_min=1e-6
    ).field(pts)
    rel = np.linalg.norm(coarse - fine, axis=1) / np.maximum(
        np.linalg.norm(fine, axis=1), 1e-300
    )
    return {"d": d, "max_rel_err_sweep_grid_vs_fine": sig(float(rel.max()))}


# --------------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------------- #
def render_markdown(R: dict[str, Any]) -> str:
    L: list[str] = []
    L.append(
        "### Item 1-2: antipodal midpoint (schedule=trunc, unit-RMS coordinates)\n"
    )
    L.append(
        "| d | ‖x₁‖ | ‖f*(0)‖ / RMS_axis | max off-axis / RMS | ds/dα at 0⁺ (α=1e-4) | ∂²E along axis | ∂²E orthogonal (3 dirs) | E[γ ∣ x̂=0] | class |"
    )
    L.append("| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |")
    for r in R["midpoint"]:
        L.append(
            f"| {r['d']} | {r['norm_x1']:.4g} | {r['f0_over_rms']:.3g} | {r['max_off_axis_component_over_rms']:.2g} | "
            f"{r['ds_dalpha_at_0plus']['0.0001']:.5g} | {r['d2E_along_axis']:.5g} | "
            f"{', '.join(f'{e:.4g}' for e in r['d2E_orthogonal'])} | {r['posterior_gamma_mean_at_midpoint']:.3g} | {r['classification']} |"
        )
    L.append("")
    L.append("Secant slopes `s(α)/α` (`f(αx₁) = s(α)·x₁`):\n")
    L.append("| d | α=1e-4 | α=1e-3 | α=1e-2 | s(0.25) | s(0.5) | s(0.75) | s(1.0) |")
    L.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in R["midpoint"]:
        sp = r["s_profile"]
        L.append(
            f"| {r['d']} | {r['ds_dalpha_at_0plus']['0.0001']:.5g} | {r['ds_dalpha_at_0plus']['0.001']:.5g} | "
            f"{r['ds_dalpha_at_0plus']['0.01']:.5g} | {sp['0.250']:.4g} | {sp['0.500']:.4g} | {sp['0.750']:.4g} | {sp['1.000']:.3g} |"
        )
    L.append("")
    if R.get("midpoint_linear"):
        L.append("Same, `c_linear` schedule (f = λ(x̂ − E[x∣x̂])):\n")
        L.append(
            "| d | ‖f*(0)‖ / RMS | ds/dα at 0⁺ | ∂²E axis | ∂²E orth (mean) | class |"
        )
        L.append("| ---: | ---: | ---: | ---: | ---: | --- |")
        for r in R["midpoint_linear"]:
            L.append(
                f"| {r['d']} | {r['f0_over_rms']:.3g} | {r['ds_dalpha_at_0plus']['0.0001']:.5g} | {r['d2E_along_axis']:.5g} | "
                f"{np.mean(r['d2E_orthogonal']):.4g} | {r['classification']} |"
            )
        L.append("")
    L.append(
        "### Item 3: paper GD (η=%g, %d steps) from N(0,I) inits, adaptive stop ‖f‖ < g_min\n"
        % (PAPER_ETA, PAPER_STEPS)
    )
    L.append(
        "| d | n_traj | ‖f‖ at init (mean) | ‖f‖ final (mean) | min ‖f‖ over all steps | ever in midpoint ball | final dist to data / ‖x₁‖ |"
    )
    L.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in R["gd_montecarlo"]:
        L.append(
            f"| {r['d']} | {r['n_traj']} | {r['fnorm_at_init_mean']:.4g} | {r['fnorm_final_mean']:.4g} | {r['min_fnorm_over_trajectories']:.4g} | "
            f"{r['frac_ever_in_midpoint_ball']:.3g} | {r['final_dist_to_data_mean_over_norm']:.3g} |"
        )
    L.append("")
    L.append(
        "Termination outcome by threshold (`g_min = ρ·λ·‖x₁‖`; ρ=0.04 reproduces the paper's g_min=10 at d=4096):\n"
    )
    L.append(
        "| d | ρ | g_min | terminated | in midpoint ball (r=0.1‖x₁‖) | near data | elsewhere | mean stop step |"
    )
    L.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in R["gd_montecarlo"]:
        for b in r["by_gmin"]:
            ms = "—" if b["mean_stop_step"] is None else f"{b['mean_stop_step']:.1f}"
            L.append(
                f"| {r['d']} | {b['rho']} | {b['g_min']:.4g} | {b['frac_terminated']:.3g} | {b['frac_terminated_in_midpoint_ball']:.3g} | "
                f"{b['frac_terminated_near_data']:.3g} | {b['frac_terminated_elsewhere']:.3g} | {ms} |"
            )
    L.append("")
    L.append(
        "### Item 4: residual field at training points (16 random unit-RMS points)\n"
    )
    L.append(
        "| d | γ_max = 1−u_min | RMS ‖f‖ at N(0,I) | ‖f*(x_i)‖/RMS (mean) | ‖f*(x_i)‖/RMS (max) | ‖f*(x_i)‖ abs max | ‖f‖/RMS at x_i + 0.05‖x_i‖ξ | radial stiffness there |"
    )
    L.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in R["training_points"]:
        for b in r["by_u_min"]:
            L.append(
                f"| {r['d']} | {b['gamma_max']:.6g} | {b['rms_f_at_N01_inits']:.4g} | {b['f_at_training_points_over_rms_mean']:.3g} | "
                f"{b['f_at_training_points_over_rms_max']:.3g} | {b['f_at_training_points_abs_max']:.3g} | "
                f"{b['f_at_probe_over_rms_mean']:.3g} | {b['probe_radial_stiffness_mean']:.4g} |"
            )
    L.append("")
    fit = R["training_points_fit"]

    def fit_str(k: str) -> str:
        f = fit[k]
        sl = "not fittable" if f["loglog_slope"] is None else f"{f['loglog_slope']:.3g}"
        return f"slope {sl} ({f['n_nonzero']} nonzero d; exactly 0 at d={f['exact_zero_at_d']})"

    L.append(
        f"Log-log fit of ‖f*(x_i)‖/RMS vs d — u_min=1e-2: **{fit_str('umin_1e-2')}**; "
        f"u_min=1e-3: **{fit_str('umin_1e-3')}**; probe 0.05‖x_i‖ off the datum: **{fit_str('probe')}**. "
        f"Hypothesis N7 predicts slope −0.5.\n"
    )
    L.append(
        "### Item 5: equilateral 3-point dataset — stationary points off the data\n"
    )
    L.append(
        "| d | R | point (plane coords / R) | ‖f‖/RMS | stationary | dist to datum / R | in-plane Hessian eigs | orth ∂²E | type |"
    )
    L.append("| ---: | ---: | --- | ---: | --- | ---: | --- | ---: | --- |")
    for r in R["equilateral"]:
        for s in r["stationary_points"]:
            o = "n/a" if s["orthogonal_d2E"] is None else f"{s['orthogonal_d2E']:.4g}"
            L.append(
                f"| {r['d']} | {r['radius']:.3g} | ({s['point_plane_coords_over_radius'][0]:.4g}, {s['point_plane_coords_over_radius'][1]:.4g}) | "
                f"{s['fnorm_over_rms']:.2g} | {s['is_stationary']} | {s['dist_to_nearest_datum_over_radius']:.3g} | "
                f"{s['inplane_hessian_eigs'][0]:.4g}, {s['inplane_hessian_eigs'][1]:.4g} | {o} | {s['type']} |"
            )
    L.append("")
    L.append(
        "| d | R | spurious strict minima | GD endpoints within 0.1R of a datum | within 0.1R of centroid | max endpoint dist to datum / R |"
    )
    L.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in R["equilateral"]:
        e = r["gd_endpoints"]
        L.append(
            f"| {r['d']} | {r['radius']:.3g} | {r['n_spurious_strict_minima']} | {e['frac_within_0p1R_of_datum']:.3g} | "
            f"{e['frac_within_0p1R_of_centroid']:.3g} | {e['max_dist_to_datum_over_radius']:.3g} |"
        )
    L.append("")
    L.append(
        "### Quadrature self-check (sweep grid 1024+128 nodes, u_min=1e-3 vs 32768+4096, u_min=1e-6; antipodal, 17 probe points)\n"
    )
    L.append("| d | max rel. error |")
    L.append("| ---: | ---: |")
    for q in R["quadrature_check"]:
        L.append(f"| {q['d']} | {q['max_rel_err_sweep_grid_vs_fine']:.2g} |")
    L.append("")
    L.append(f"Wall time: {R['wall_seconds']:.1f} s (seed {SEED}).")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--schedule", default="trunc", choices=["trunc", "linear"])
    ap.add_argument(
        "--n-traj",
        type=int,
        default=1024,
        help="GD trajectories per d (d=4096 uses a quarter)",
    )
    ap.add_argument("--quick", action="store_true", help="skip d=4096 (smoke)")
    ap.add_argument(
        "--stem", default="eqm-oracle-field-20260902", help="output file stem"
    )
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    dims_main = [2, 64] if args.quick else [2, 64, 4096]
    dims_tp = [2, 8, 64, 512] if args.quick else [2, 8, 64, 512, 4096]
    R: dict[str, Any] = {
        "schema": "eqm_oracle_field_analysis/v1",
        "claim_class": "analysis",
        "honesty": "external_paper_rederivation_not_repo_model_claim",
        "seed": SEED,
        "paper": {
            "arxiv": "2510.02300",
            "schedule": args.schedule,
            "trunc_a": PAPER_TRUNC_A,
            "lambda": PAPER_LAMBDA,
            "eta": PAPER_ETA,
            "steps": PAPER_STEPS,
            "g_min_latent": PAPER_GMIN_LATENT,
        },
        "field_definition": "f*(xh) = lam * E[c(g)/(1-g) * (xh - x_i) | x_g = xh], posterior over (i,g) ∝ exp(-|xh - g x_i|^2/(2(1-g)^2) - d log(1-g)), g ~ U(0,1-u_min) on a log-spaced u=1-g trapezoid grid",
    }

    def tick(label: str) -> None:
        print(f"[{time.time() - t0:6.1f}s] {label}", file=sys.stderr)

    R["midpoint"] = [
        item_midpoint(d, rng, schedule=args.schedule, scale=None) for d in dims_main
    ]
    tick("midpoint")
    R["midpoint_linear"] = [
        item_midpoint(d, rng, schedule="linear", scale=None) for d in dims_main
    ]
    tick("midpoint_linear")
    rhos = (0.01, 0.04, 0.1, 0.3)
    R["gd_montecarlo"] = [
        item_gd_montecarlo(
            d,
            rng,
            n_traj=args.n_traj if d <= 64 else max(args.n_traj // 4, 64),
            rhos=rhos,
            eta=PAPER_ETA,
            steps=PAPER_STEPS,
            schedule=args.schedule,
            scale=None,
        )
        for d in dims_main
    ]
    tick("gd_montecarlo")
    u_mins = (1e-2, 1e-3)
    R["training_points"] = [
        item_training_points(
            d, rng, schedule=args.schedule, u_mins=u_mins, probe_radius_rel=0.05
        )
        for d in dims_tp
    ]
    tick("training_points")

    # log-log slopes vs d
    def slope(key: str, idx: int) -> dict[str, Any]:
        pairs = [(r["d"], r["by_u_min"][idx][key]) for r in R["training_points"]]
        nz = [(d_, y) for d_, y in pairs if y > 0]
        zero_dims = [d_ for d_, y in pairs if y == 0]
        fit = None
        if len(nz) >= 3:
            fit = sig(
                float(
                    np.polyfit(
                        np.log([d_ for d_, _ in nz]), np.log([y for _, y in nz]), 1
                    )[0]
                )
            )
        return {"loglog_slope": fit, "n_nonzero": len(nz), "exact_zero_at_d": zero_dims}

    R["training_points_fit"] = {
        "umin_1e-2": slope("f_at_training_points_over_rms_mean", 0),
        "umin_1e-3": slope("f_at_training_points_over_rms_mean", 1),
        "probe": slope("f_at_probe_over_rms_mean", 1),
        "hypothesis_N7_predicted_slope": -0.5,
        "analytic": "at xh = x_i the i-th term has weight ∝ u^{-d} (non-integrable at u=0): the posterior is a point mass at (i, gamma=1) whose target g·(x_i − x_i) = 0, so f*(x_i) = 0 exactly for every d; with a cutoff u_min the i-th term keeps mass ~u_min^(1-d) while the j≠i terms keep O(1) mass, so the residual scales like u_min^(d-1) (d=2 check: 10x smaller cutoff gives 0.107x, predicted 0.1x)",
    }
    R["equilateral"] = [
        item_equilateral(2, rng, radius=math.sqrt(2), schedule=args.schedule),
        item_equilateral(2, rng, radius=6.0, schedule=args.schedule),
        item_equilateral(64, rng, radius=8.0, schedule=args.schedule),
    ]
    tick("equilateral")
    R["quadrature_check"] = [quadrature_check(d, rng, args.schedule) for d in dims_main]
    tick("quadrature_check")
    R["wall_seconds"] = round(time.time() - t0, 1)
    R["version_stamp"] = build_version_stamp(*STAMP_COMPONENTS)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.stem}.json").write_text(
        json.dumps(R, indent=2) + "\n", encoding="utf-8"
    )
    md = render_markdown(R)
    (args.out / f"{args.stem}.tables.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
