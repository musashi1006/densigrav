"""Publication-quality 2-D gravity section figure (gravity fit + density model).

The calculated curve and misfit statistics are recomputed from the model
polygon, so the figure is always consistent with the model file (it never
relies on a possibly-stale ``talwani_pred_mgal`` column).

matplotlib is imported lazily so the package stays importable without the
optional ``viz`` extra.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import yaml

from .talwani2d import talwani_gz_polygon

KM = 1e-3


def load_model(path: Path) -> tuple[float, np.ndarray]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    drho = float(data["density_contrast_kgm3"])
    verts = np.asarray(data["vertices_xz_m"], dtype=float)
    if verts.ndim != 2 or verts.shape[1] != 2 or verts.shape[0] < 3:
        raise ValueError("vertices_xz_m must be a list of >=3 [x, z] pairs")
    return drho, verts


def outline_with_data_extent(ax, verts, x_data_lo, x_data_hi, *, color, lw, zorder, label=None):
    """Draw a polygon outline: solid where covered by data, dashed where not.

    The dashed portions mark geometry that lies outside the gravity coverage
    and is therefore unconstrained (geological-section convention for
    inferred boundaries).
    """
    vc = np.vstack([verts, verts[:1]])
    labeled = False
    for i in range(len(vc) - 1):
        xs = np.linspace(vc[i, 0], vc[i + 1, 0], 64)
        zs = np.linspace(vc[i, 1], vc[i + 1, 1], 64)
        inside = (xs >= x_data_lo) & (xs <= x_data_hi)
        for mask, ls in ((inside, "-"), (~inside, "--")):
            if not mask.any():
                continue
            lab = label if (label is not None and not labeled and ls == "-") else None
            if lab is not None:
                labeled = True
            ax.plot(
                np.where(mask, xs, np.nan) * KM,
                np.where(mask, zs, np.nan) * KM,
                color=color,
                lw=lw,
                ls=ls,
                zorder=zorder,
                label=lab,
            )


def shade_no_data(axes, x_lo, x_hi, x_data_lo, x_data_hi, *, label_ax=None):
    """Grey out the parts of the section without gravity coverage."""
    for ax in axes:
        lab = "No gravity data" if ax is label_ax else None
        if x_data_hi < x_hi:
            ax.axvspan(
                x_data_hi * KM, x_hi * KM, color="0.55", alpha=0.13, lw=0, zorder=1.5, label=lab
            )
            lab = None
        if x_lo < x_data_lo:
            ax.axvspan(
                x_lo * KM, x_data_lo * KM, color="0.55", alpha=0.13, lw=0, zorder=1.5, label=lab
            )


def plot_section_model(
    model: Path,
    profile: Path,
    out: Path,
    *,
    value_col: str = "residual_mgal",
    section_name: str = "Section",
    exclude_dist: Iterable[float] = (),
    exclude_tol: float = 2.0,
    obs_height: str = "sealevel",
    equal_aspect: bool = True,
) -> dict:
    """Render the 2-panel section figure (png + sibling pdf). Returns misfit stats."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    model, profile, out = Path(model), Path(profile), Path(out)
    drho, verts = load_model(model)

    df = pd.read_csv(profile)
    if "dist_m" not in df.columns or value_col not in df.columns:
        raise ValueError(f"profile must contain 'dist_m' and '{value_col}'")

    dist = df["dist_m"].to_numpy(dtype=float)
    obs = df[value_col].to_numpy(dtype=float)
    elev = df["elev_m"].to_numpy(dtype=float) if "elev_m" in df.columns else np.zeros_like(dist)

    excl = np.zeros(len(dist), dtype=bool)
    for xd in exclude_dist:
        excl |= np.abs(dist - xd) <= exclude_tol
    keep = ~excl

    z_obs = (-elev) if obs_height == "elev" else np.zeros_like(dist)

    # recompute calculated gravity from the model (stations + dense curve)
    pred_st = talwani_gz_polygon(dist, z_obs, verts, drho)
    res = obs - pred_st
    rms = float(np.sqrt(np.mean(res[keep] ** 2)))
    vr = float(100.0 * (1.0 - np.var(res[keep]) / np.var(obs[keep])))

    xpad = 0.04 * (dist.max() - dist.min())
    x_lo = min(dist.min(), verts[:, 0].min()) - xpad
    x_hi = max(dist.max(), verts[:, 0].max()) + xpad
    x_data_lo = float(dist[keep].min())
    x_data_hi = float(dist[keep].max())
    xd = np.linspace(x_lo, x_hi, 500)
    if obs_height == "elev":
        o = np.argsort(dist)
        zd = -np.interp(xd, dist[o], elev[o])
    else:
        zd = np.zeros_like(xd)
    pred_dense = talwani_gz_polygon(xd, zd, verts, drho)

    plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.8})
    if equal_aspect:
        figsize, ratios = (7.5, 4.8), [1.0, 0.7]
    else:
        figsize, ratios = (7.5, 6.4), [1.0, 1.5]
    fig, (axg, axm) = plt.subplots(
        2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": ratios}
    )

    # (top) gravity panel
    axg.plot(xd * KM, pred_dense, "-", color="#c0392b", lw=2.0, zorder=3, label="Calculated")
    axg.scatter(
        dist[keep] * KM,
        obs[keep],
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
    axg.legend(loc="upper left", fontsize=8, framealpha=0.92)
    stats = (
        f"$\\Delta\\rho$ = {drho:+.0f} kg m$^{{-3}}$\n"
        f"RMS misfit = {rms:.2f} mGal\n"
        f"Variance reduction = {vr:.0f}%\n"
        f"N = {int(keep.sum())}"
    )
    axg.text(
        0.985,
        0.06,
        stats,
        transform=axg.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.92),
    )
    axg.set_title(f"{section_name}  —  2-D gravity model", fontsize=11, loc="left")

    # (bottom) model panel
    order = np.argsort(dist)
    topo_z = -elev[order] * KM
    axm.fill_between(dist[order] * KM, topo_z, 0.0, color="#efe7dd", zorder=0)
    axm.plot(dist[order] * KM, topo_z, color="#6e4b3a", lw=1.4, zorder=5, label="Ground surface")

    body_color = "#d9544d" if drho > 0 else "#3b7dd8"
    edge_color = "#922b21" if drho > 0 else "#1f4e79"
    axm.fill(
        verts[:, 0] * KM,
        verts[:, 1] * KM,
        facecolor=body_color,
        alpha=0.40,
        edgecolor="none",
        zorder=3,
    )
    outline_with_data_extent(axm, verts, x_data_lo, x_data_hi, color=edge_color, lw=2.0, zorder=4)
    if not equal_aspect:
        # in-body label only fits when the panel is vertically stretched
        cx = float(verts[:, 0].mean()) * KM
        cz = float(verts[:, 1].mean()) * KM
        axm.text(
            cx,
            cz,
            f"$\\Delta\\rho$ = {drho:+.0f}\nkg m$^{{-3}}$",
            ha="center",
            va="center",
            fontsize=9.5,
            color=edge_color,
            zorder=6,
        )
    axm.axhline(0.0, color="#34607d", lw=0.9, ls="--", zorder=2)
    axm.text(x_lo * KM, 0.0, " sea level", color="#34607d", fontsize=7.5, va="bottom", ha="left")

    axm.set_ylabel("Depth below sea level (km)")
    axm.set_xlabel("Distance along section (km)")
    axm.grid(alpha=0.25)
    axm.set_xlim(x_lo * KM, x_hi * KM)

    zmax = float(verts[:, 1].max()) * KM
    emax = float(elev.max()) * KM
    axm.set_ylim(-(emax * 1.5 + 0.05), zmax * 1.10)
    axm.invert_yaxis()
    if (x_hi - x_data_hi) * KM > 0.4:
        axm.text(
            0.5 * (x_data_hi + x_hi) * KM,
            0.22 * zmax,
            "no data",
            color="0.35",
            fontsize=7.5,
            ha="center",
            va="center",
            style="italic",
            zorder=7,
        )

    if equal_aspect:
        axm.set_aspect("equal", adjustable="box")
        ve_note = "no vertical exaggeration (1:1)"
    else:
        bbox = axm.get_position()
        w_in = fig.get_figwidth() * bbox.width
        h_in = fig.get_figheight() * bbox.height
        dx = (x_hi - x_lo) * KM
        dz = (zmax * 1.10) + (emax * 1.5 + 0.05)
        # VE = horizontal scale / vertical scale (>1 means vertically stretched)
        ve = (dx / w_in) / (dz / h_in)
        ve_note = f"vertical exaggeration ≈ {ve:.1f}×"
    axm.text(
        0.985,
        0.04,
        ve_note,
        transform=axm.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="0.35",
        style="italic",
    )

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    pdf = out.with_suffix(".pdf")
    fig.savefig(pdf)
    plt.close(fig)
    return {
        "rms_mgal": rms,
        "variance_reduction_pct": vr,
        "n": int(keep.sum()),
        "out": str(out),
        "pdf": str(pdf),
    }
