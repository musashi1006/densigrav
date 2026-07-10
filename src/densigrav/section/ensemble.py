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

from .plotting import load_model, outline_with_data_extent, shade_no_data
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
    drho_sigma: float = 0.0,
    n: int = 400,
    accept_factor: float = 1.2,
    seed: int = 0,
    obs_height: str = "elev",
    equal_aspect: bool = True,
) -> dict:
    """Run the ensemble, render the uncertainty figure, and write the parameter CSV.

    With ``obs_height="elev"`` (default) stations are evaluated at their real
    height (z_obs = -elev_m; z is positive down), consistent with
    ``plot_section_model``; ``"sealevel"`` evaluates every station at z=0.
    """
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

    if obs_height not in ("elev", "sealevel"):
        raise ValueError("obs_height must be 'elev' or 'sealevel'")
    if obs_height == "elev" and "elev_m" not in df.columns:
        raise ValueError(
            "obs_height='elev' requires an elev_m column in profile.csv "
            "(pass obs_height='sealevel' / --no-use-elev to model observations at z=0)"
        )

    excl = np.zeros(len(dist), dtype=bool)
    for xd in exclude_dist:
        excl |= np.abs(dist - xd) <= exclude_tol
    keep = ~excl
    xk, yk = dist[keep], obs[keep]
    # stations observe at their real height (z is positive down, so z_obs = -elev)
    zk = (-elev[keep]) if obs_height == "elev" else np.zeros_like(xk)

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
    raw_drho = np.empty(n)
    rms_obs = np.empty(n)
    for k in range(n):
        yp = yk + rng.normal(0.0, sigma, size=yk.shape)
        # propagate measured density-contrast uncertainty (petrophysical 1-sigma)
        drho_k = drho if drho_sigma <= 0 else max(10.0, drho + rng.normal(0.0, drho_sigma))
        raw_drho[k] = drho_k
        start = np.clip(p0_best * (1.0 + rng.normal(0.0, 0.25, size=4)), lo, hi)
        try:
            raw[k] = _fit_trapezoid(least_squares, xk, yp, zk, drho_k, start, lo, hi)
        except Exception:
            raw[k] = best_p
        rms_obs[k] = float(
            np.sqrt(np.mean((yk - talwani_gz_polygon(xk, zk, trapezoid(*raw[k]), drho_k)) ** 2))
        )

    accepted = rms_obs <= accept_factor * best_rms
    params = raw[accepted]
    drhos = raw_drho[accepted]
    n_acc = int(accepted.sum())
    if n_acc < 10:
        raise ValueError(f"Too few acceptable models ({n_acc}); loosen accept_factor or raise n")

    out.parent.mkdir(parents=True, exist_ok=True)
    ens_csv = out.parent / (out.stem + "_params.csv")
    df_out = pd.DataFrame(
        params, columns=["x0_m", "halfwidth_top_m", "halfwidth_base_m", "base_depth_m"]
    )
    df_out["drho_kgm3"] = drhos
    df_out.to_csv(ens_csv, index=False)

    # predictive gravity envelope on a dense grid
    xpad = 0.04 * (dist.max() - dist.min())
    x_lo = min(dist.min(), best_verts[:, 0].min()) - xpad
    x_hi = max(dist.max(), best_verts[:, 0].max()) + xpad
    xd = np.linspace(x_lo, x_hi, 400)
    if obs_height == "elev":
        o = np.argsort(dist)
        zd = -np.interp(xd, dist[o], elev[o])
    else:
        zd = np.zeros_like(xd)
    curves = np.empty((n_acc, xd.size))
    for k in range(n_acc):
        curves[k] = talwani_gz_polygon(xd, zd, trapezoid(*params[k]), drhos[k])
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
    # mask where gravity data provide no constraint (outside station coverage):
    # there the ensemble is bounded only by the parameter priors, and painting
    # it would contradict the "no data" shading
    x_data_lo, x_data_hi = float(xk.min()), float(xk.max())
    outside = (GX < x_data_lo) | (GX > x_data_hi)
    prob_masked = np.ma.masked_where((prob < 0.05) | outside, prob)
    prob_contour = np.ma.masked_where(outside, prob)

    # ---------------- figure ----------------
    plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.8})
    if equal_aspect:
        figsize, ratios = (7.9, 5.0), [1.0, 0.7]
    else:
        figsize, ratios = (7.5, 6.6), [1.0, 1.5]
    fig, (axg, axm) = plt.subplots(
        2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": ratios}
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
    shade_no_data((axg, axm), x_lo, x_hi, x_data_lo, x_data_hi, label_ax=axg)
    axg.legend(loc="upper left", fontsize=7.5, framealpha=0.92)
    drho_note = (
        f"$\\Delta\\rho$ = {drho:+.0f} $\\pm$ {drho_sigma:.0f} kg m$^{{-3}}$ (measured)"
        if drho_sigma > 0
        else f"$\\Delta\\rho$ = {drho:+.0f} kg m$^{{-3}}$ (fixed)"
    )
    stats = (
        f"{drho_note}\n"
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
        prob_contour,
        levels=[0.25, 0.5, 0.75],
        colors="#7b241c",
        linewidths=[0.6, 0.9, 0.6],
        linestyles=["dotted", "solid", "dashed"],
        zorder=4,
    )
    axm.clabel(cs, fmt={0.25: "25%", 0.5: "50%", 0.75: "75%"}, fontsize=6.5, inline=True)
    outline_with_data_extent(
        axm,
        best_verts,
        x_data_lo,
        x_data_hi,
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
    if (x_hi - x_data_hi) * KM > 0.4:
        axm.text(
            0.5 * (x_data_hi + x_hi) * KM,
            0.5 * zb_hi * KM,
            "no data",
            color="0.35",
            fontsize=7.5,
            ha="center",
            va="center",
            style="italic",
            zorder=7,
        )
    axm.legend(loc="lower left", fontsize=7.5, framealpha=0.92)

    cbar = fig.colorbar(pcm, ax=axm, pad=0.012, fraction=0.045)
    cbar.set_label("Model support\nP(inside body)", fontsize=8)
    cbar.ax.tick_params(labelsize=7.5)

    if equal_aspect:
        axm.set_aspect("equal", adjustable="box")
        ve_note = "no vertical exaggeration (1:1)"
    else:
        bbox = axm.get_position()
        w_in = fig.get_figwidth() * bbox.width
        h_in = fig.get_figheight() * bbox.height
        dx = (x_hi - x_lo) * KM
        dz = (zb_hi * KM) + (emax * 1.5 + 0.05)
        # VE = horizontal scale / vertical scale (>1 means vertically stretched)
        ve = (dx / w_in) / (dz / h_in)
        ve_note = f"vertical exaggeration ≈ {ve:.1f}×"
    axm.text(
        0.985,
        0.06,
        ve_note,
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
        "drho_kgm3": (
            float(np.median(drhos)),
            float(np.percentile(drhos, 5)),
            float(np.percentile(drhos, 95)),
        ),
    }
