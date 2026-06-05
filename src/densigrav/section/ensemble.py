"""Acceptable-model ensemble for a 2-D gravity section (honest non-uniqueness).

Monte-Carlo error propagation: perturb the observed anomaly with Gaussian
noise, re-fit a surface-topped trapezoid from randomized starts, and keep the
solutions that fit the observed data essentially as well as the best model.
The accepted family is summarized as a body-presence probability map and a
5-95 % predictive envelope of the calculated anomaly.

scipy / matplotlib are imported lazily so the package stays importable without
the optional ``grid`` / ``viz`` extras.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .plotting import load_model
from .talwani2d import talwani_gz_polygon

KM = 1e-3


def trapezoid(x0: float, ht: float, hb: float, zb: float, zt: float = 0.0) -> np.ndarray:
    """Surface-topped trapezoid: top half-width ht at depth zt, base half-width hb at depth zb."""
    return np.array([[x0 - ht, zt], [x0 + ht, zt], [x0 + hb, zb], [x0 - hb, zb]], dtype=float)


def _fit_trapezoid(least_squares, x, y, z, drho, p0, lo, hi):
    def res(p):
        x0, ht, hb, zb = p
        if zb <= 60.0:
            return 1e6 * np.ones_like(y)
        return talwani_gz_polygon(x, z, trapezoid(x0, ht, hb, zb), drho) - y

    return least_squares(res, p0, bounds=(lo, hi), max_nfev=400).x


def run_ensemble(
    model: Path,
    profile: Path,
    out: Path,
    *,
    value_col: str = "residual_mgal",
    section_name: str = "Section",
    exclude_dist: Iterable[float] = (),
    exclude_tol: float = 2.0,
    sigma: float = 0.8,
    n: int = 400,
    accept_factor: float = 1.2,
    seed: int = 0,
) -> dict:
    """Run the ensemble, render the uncertainty figure, and write the parameter CSV."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.path import Path as MplPath
    from scipy.optimize import least_squares

    model, profile, out = Path(model), Path(profile), Path(out)
    drho, best_verts = load_model(model)

    df = pd.read_csv(profile)
    dist = df["dist_m"].to_numpy(dtype=float)
    obs = df[value_col].to_numpy(dtype=float)
    elev = df["elev_m"].to_numpy(dtype=float) if "elev_m" in df.columns else np.zeros_like(dist)

    excl = np.zeros(len(dist), dtype=bool)
    for xd in exclude_dist:
        excl |= np.abs(dist - xd) <= exclude_tol
    keep = ~excl
    xk, yk = dist[keep], obs[keep]
    zk = np.zeros_like(xk)

    # parameter bounds (geological priors): top fixed at sea level (z=0)
    xmin, xmax = float(dist.min()), float(dist.max())
    lo = np.array([xmin + 200.0, 200.0, 100.0, 200.0])
    hi = np.array([xmax + 800.0, 3500.0, 3500.0, 4000.0])

    rng = np.random.default_rng(seed)

    def _rms(p):
        return float(np.sqrt(np.mean((yk - talwani_gz_polygon(xk, zk, trapezoid(*p), drho)) ** 2)))

    # robust best fit via several starts (the misfit surface is non-convex; a single
    # generic start can converge to a poor local minimum)
    x0_peak = float(xk[np.argmax(np.abs(yk))])
    start_centers = [
        x0_peak,
        0.5 * (xmin + xmax),
        xmin + 0.25 * (xmax - xmin),
        xmin + 0.75 * (xmax - xmin),
    ]
    best_p, best_rms = None, np.inf
    for xc in start_centers:
        p0 = np.clip(np.array([xc, 1500.0, 1000.0, 1500.0]), lo, hi)
        cand = _fit_trapezoid(least_squares, xk, yk, zk, drho, p0, lo, hi)
        r = _rms(cand)
        if r < best_rms:
            best_rms, best_p = r, cand
    p0_best = best_p.copy()

    raw = np.empty((n, 4))
    rms_obs = np.empty(n)
    for k in range(n):
        yp = yk + rng.normal(0.0, sigma, size=yk.shape)
        start = np.clip(p0_best * (1.0 + rng.normal(0.0, 0.25, size=4)), lo, hi)
        try:
            raw[k] = _fit_trapezoid(least_squares, xk, yp, zk, drho, start, lo, hi)
        except Exception:
            raw[k] = best_p
        rms_obs[k] = _rms(raw[k])

    accepted = rms_obs <= accept_factor * best_rms
    params = raw[accepted]
    n_acc = int(accepted.sum())
    if n_acc < 10:
        raise ValueError(f"Too few acceptable models ({n_acc}); loosen accept_factor or raise n")

    out.parent.mkdir(parents=True, exist_ok=True)
    ens_csv = out.parent / (out.stem + "_params.csv")
    pd.DataFrame(
        params, columns=["x0_m", "halfwidth_top_m", "halfwidth_base_m", "base_depth_m"]
    ).to_csv(ens_csv, index=False)

    # predictive gravity envelope on a dense grid
    xpad = 0.04 * (dist.max() - dist.min())
    x_lo = min(dist.min(), best_verts[:, 0].min()) - xpad
    x_hi = max(dist.max(), best_verts[:, 0].max()) + xpad
    xd = np.linspace(x_lo, x_hi, 400)
    zd = np.zeros_like(xd)
    curves = np.empty((n_acc, xd.size))
    for k in range(n_acc):
        curves[k] = talwani_gz_polygon(xd, zd, trapezoid(*params[k]), drho)
    q05, q95 = np.percentile(curves, [5, 95], axis=0)
    best_curve = talwani_gz_polygon(xd, zd, trapezoid(*best_p), drho)

    pred_best_st = talwani_gz_polygon(xk, zk, trapezoid(*best_p), drho)
    rms = float(np.sqrt(np.mean((yk - pred_best_st) ** 2)))
    vr = float(100.0 * (1.0 - np.var(yk - pred_best_st) / np.var(yk)))

    # body-presence probability on an (x,z) grid
    zb_hi = float(np.percentile(params[:, 3], 99)) * 1.08
    gx = np.linspace(x_lo, x_hi, 240)
    gz = np.linspace(0.0, zb_hi, 150)
    GX, GZ = np.meshgrid(gx, gz)
    pts = np.column_stack([GX.ravel(), GZ.ravel()])
    inside = np.zeros(pts.shape[0], dtype=float)
    for k in range(n_acc):
        inside += MplPath(trapezoid(*params[k])).contains_points(pts)
    prob = (inside / n_acc).reshape(GX.shape)
    prob_masked = np.ma.masked_less(prob, 0.05)

    # ---------------- figure ----------------
    plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.8})
    fig, (axg, axm) = plt.subplots(
        2, 1, figsize=(7.5, 6.6), sharex=True, gridspec_kw={"height_ratios": [1.0, 1.5]}
    )

    axg.fill_between(
        xd * KM,
        q05,
        q95,
        color="#c0392b",
        alpha=0.20,
        lw=0,
        label="Calculated (5–95 % of ensemble)",
        zorder=2,
    )
    axg.plot(xd * KM, best_curve, "-", color="#c0392b", lw=2.0, zorder=3, label="Best-fit model")
    axg.scatter(
        xk * KM,
        yk,
        s=44,
        facecolor="#2c3e50",
        edgecolor="white",
        linewidth=0.6,
        zorder=4,
        label="Observed (residual)",
    )
    if excl.any():
        axg.scatter(
            dist[excl] * KM,
            obs[excl],
            s=48,
            facecolor="none",
            edgecolor="#95a5a6",
            linewidth=1.3,
            zorder=4,
            label="Excluded outlier",
        )
    axg.axhline(0.0, color="0.6", lw=0.8)
    axg.set_ylabel("Residual Bouguer\nanomaly (mGal)")
    axg.grid(alpha=0.25)
    axg.legend(loc="upper left", fontsize=7.5, framealpha=0.92)
    stats = (
        f"$\\Delta\\rho$ = {drho:+.0f} kg m$^{{-3}}$ (fixed)\n"
        f"RMS misfit = {rms:.2f} mGal\n"
        f"Variance reduction = {vr:.0f}%\n"
        f"noise $\\sigma$ = {sigma:.1f} mGal,  N = {int(keep.sum())}"
    )
    axg.text(
        0.985,
        0.06,
        stats,
        transform=axg.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.92),
    )
    axg.set_title(f"{section_name}  —  2-D gravity model with uncertainty", fontsize=11, loc="left")

    order = np.argsort(dist)
    topo_z = -elev[order] * KM
    axm.fill_between(dist[order] * KM, topo_z, 0.0, color="#efe7dd", zorder=0)
    axm.plot(dist[order] * KM, topo_z, color="#6e4b3a", lw=1.4, zorder=6, label="Ground surface")

    pcm = axm.pcolormesh(
        gx * KM, gz * KM, prob_masked, cmap="YlOrRd", vmin=0.0, vmax=1.0, shading="auto", zorder=2
    )
    cs = axm.contour(
        GX * KM,
        GZ * KM,
        prob,
        levels=[0.25, 0.5, 0.75],
        colors="#7b241c",
        linewidths=[0.6, 0.9, 0.6],
        linestyles=["dotted", "solid", "dashed"],
        zorder=4,
    )
    axm.clabel(cs, fmt={0.25: "25%", 0.5: "50%", 0.75: "75%"}, fontsize=6.5, inline=True)
    axm.plot(
        np.append(best_verts[:, 0], best_verts[0, 0]) * KM,
        np.append(best_verts[:, 1], best_verts[0, 1]) * KM,
        color="#1b2631",
        lw=1.8,
        zorder=5,
        label="Best-fit body",
    )
    axm.axhline(0.0, color="#34607d", lw=0.9, ls="--", zorder=3)
    axm.text(x_lo * KM, 0.0, " sea level", color="#34607d", fontsize=7.5, va="bottom", ha="left")

    axm.set_ylabel("Depth below sea level (km)")
    axm.set_xlabel("Distance along section (km)")
    axm.grid(alpha=0.2)
    axm.set_xlim(x_lo * KM, x_hi * KM)
    emax = float(elev.max()) * KM
    axm.set_ylim(-(emax * 1.5 + 0.05), zb_hi * KM)
    axm.invert_yaxis()
    axm.legend(loc="lower left", fontsize=7.5, framealpha=0.92)

    cbar = fig.colorbar(pcm, ax=axm, pad=0.012, fraction=0.045)
    cbar.set_label("Model support\nP(inside body)", fontsize=8)
    cbar.ax.tick_params(labelsize=7.5)

    bbox = axm.get_position()
    w_in = fig.get_figwidth() * bbox.width
    h_in = fig.get_figheight() * bbox.height
    dx = (x_hi - x_lo) * KM
    dz = (zb_hi * KM) + (emax * 1.5 + 0.05)
    ve = (dz / h_in) / (dx / w_in)
    axm.text(
        0.985,
        0.04,
        f"vertical exaggeration ≈ {ve:.1f}×",
        transform=axm.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="0.35",
        style="italic",
    )

    fig.tight_layout()
    fig.savefig(out, dpi=300)
    pdf = out.with_suffix(".pdf")
    fig.savefig(pdf)
    plt.close(fig)

    def _ci(col):
        return (
            float(np.median(params[:, col])),
            float(np.percentile(params[:, col], 5)),
            float(np.percentile(params[:, col], 95)),
        )

    return {
        "out": str(out),
        "pdf": str(pdf),
        "params_csv": str(ens_csv),
        "n_draws": int(n),
        "n_accepted": n_acc,
        "best_rms_mgal": best_rms,
        "rms_mgal": rms,
        "variance_reduction_pct": vr,
        "best_base_depth_m": float(best_p[3]),
        "best_center_x0_m": float(best_p[0]),
        "center_x0_m": _ci(0),
        "base_depth_m": _ci(3),
        "halfwidth_top_m": _ci(1),
        "halfwidth_base_m": _ci(2),
    }
