#!/usr/bin/env python3
"""End-member hypothesis test: shallow tabular sheet (H1) vs steep compact body (H2).

Geological question
-------------------
The contact aureole around the pluton is unusually wide. Two end-member
explanations:

  H1 "shallow tabular sheet": the pluton continues underground as a thin,
     shallow, laterally extensive sheet beneath the aureole, so the heat
     source is simply close below -> no special geotherm needed.
  H2 "steep compact body": the pluton walls are steep and the body is
     compact -> the wide aureole requires an elevated geotherm at
     intrusion time.

Both families have comparable parameter counts, so best-fit RMS / AIC are
directly comparable. If the mapped aureole interval is supplied
(--aureole-from/--aureole-to, meters along the section), the sheet is required
to underlie it (that is H1's raison d'etre) and an additional thickness scan
reports the maximum sheet thickness admissible beneath the aureole.

Physics note: the interior anomaly of a wide sheet is the Bouguer slab value
2*pi*G*drho*t, independent of depth. So a sheet beneath the aureole cannot
hide by being deeper -- only by being thinner. This makes the test robust.

Observation heights use the real station elevations (z_obs = -elev_m).

Example
-------
python scripts/test_h1_h2.py \
  --profile results/2d/section_Shinshiro/profile_fit_residual_v3.csv \
  --value-col residual_mgal --exclude-dist 4305.5 --drho 78 \
  --aureole-from 4500 --aureole-to 6500 \
  --out results/figures/section_Shinshiro_h1h2.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from densigrav.section.talwani2d import talwani_gz_polygon  # noqa: E402

KM = 1e-3
G_SI = 6.67430e-11


def trapezoid(x0, ht, hb, zb, zt=0.0):
    return np.array([[x0 - ht, zt], [x0 + ht, zt], [x0 + hb, zb], [x0 - hb, zb]], float)


def sheet(xl, xr, zt, t):
    return np.array([[xl, zt], [xr, zt], [xr, zt + t], [xl, zt + t]], float)


def rms_of(pred, obs):
    return float(np.sqrt(np.mean((obs - pred) ** 2)))


def slab_mgal(drho, t):
    """Infinite-slab (interior) anomaly of a sheet of thickness t."""
    return 2.0 * np.pi * G_SI * drho * t * 1e5


def fit_h2(x, y, z, drho, top_z, least_squares):
    """Steep/compact family: surface-topped trapezoid (4 params)."""
    xmin, xmax = float(x.min()), float(x.max())
    lo = np.array([xmin + 200.0, 200.0, 100.0, max(top_z + 100.0, 200.0)])
    hi = np.array([xmax + 800.0, 3500.0, 3500.0, 4000.0])
    x0_peak = float(x[np.argmax(np.abs(y))])
    centers = [
        x0_peak,
        0.5 * (xmin + xmax),
        xmin + 0.25 * (xmax - xmin),
        xmin + 0.75 * (xmax - xmin),
    ]

    def res(p):
        x0, ht, hb, zb = p
        return talwani_gz_polygon(x, z, trapezoid(x0, ht, hb, zb, top_z), drho) - y

    best_p, best_r = None, np.inf
    for xc in centers:
        p0 = np.clip(np.array([xc, 1500.0, 1000.0, 1500.0]), lo, hi)
        sol = least_squares(res, p0, bounds=(lo, hi), max_nfev=400).x
        r = rms_of(talwani_gz_polygon(x, z, trapezoid(*sol, top_z), drho), y)
        if r < best_r:
            best_r, best_p = r, sol
    return best_p, best_r


def fit_h1(
    x,
    y,
    z,
    drho,
    elev_x,
    elev_v,
    *,
    cover,
    t_max,
    aur_from,
    aur_to,
    top_max_z,
    least_squares,
):
    """Tabular family: flat sheet (4 params: xl, xr, zt, t), kept below ground by `cover`."""
    xmin, xmax = float(x.min()), float(x.max())
    xl_lo, xl_hi = xmin - 3000.0, (aur_from if aur_from is not None else xmax - 500.0)
    xr_lo, xr_hi = (aur_to if aur_to is not None else xmin + 500.0), xmax + 4000.0
    zt_lo = cover - float(elev_v.max())
    zt_hi = 1500.0 if top_max_z is None else max(zt_lo + 1.0, float(top_max_z))
    lo = np.array([xl_lo, xr_lo, zt_lo, 10.0])
    hi = np.array([xl_hi, xr_hi, zt_hi, t_max])

    def res(p):
        xl, xr, zt, t = p
        if xr - xl < 200.0:
            return 1e6 * np.ones_like(y)
        span = (elev_x >= xl) & (elev_x <= xr)
        emin = (
            float(elev_v[span].min())
            if span.any()
            else float(np.interp(0.5 * (xl + xr), elev_x, elev_v))
        )
        pen = max(0.0, (cover - emin) - zt)  # sheet top must stay >= cover below lowest ground
        out = talwani_gz_polygon(x, z, sheet(xl, xr, zt, t), drho) - y
        return out + 0.05 * pen

    xl0s = [xl_lo + 500.0, max(xl_lo, xmin)]
    xr0s = [min(xr_hi, 0.5 * (xmin + xmax)), min(xr_hi, xmax), min(xr_hi, xmax + 2000.0)]
    zt0s = sorted({np.clip(v, zt_lo, zt_hi) for v in (zt_lo + 50.0, 0.0, 400.0)})
    best_p, best_r = None, np.inf
    for xl0 in xl0s:
        for xr0 in xr0s:
            if xr0 - xl0 < 500.0:
                continue
            for zt0 in zt0s:
                for t0 in (150.0, 450.0):
                    p0 = np.clip(np.array([xl0, xr0, zt0, t0]), lo, hi)
                    try:
                        sol = least_squares(res, p0, bounds=(lo, hi), max_nfev=400).x
                    except Exception:
                        continue
                    r = rms_of(talwani_gz_polygon(x, z, sheet(*sol), drho), y)
                    if r < best_r:
                        best_r, best_p = r, sol
    return best_p, best_r


def main() -> None:
    ap = argparse.ArgumentParser(description="H1 (tabular sheet) vs H2 (steep compact) test.")
    ap.add_argument("--profile", required=True, type=Path)
    ap.add_argument("--value-col", default="residual_mgal")
    ap.add_argument("--drho", type=float, default=78.0)
    ap.add_argument("--exclude-dist", type=float, nargs="*", default=[])
    ap.add_argument("--exclude-tol", type=float, default=2.0)
    ap.add_argument(
        "--aureole-from",
        type=float,
        default=None,
        help="western end of mapped aureole (m along section)",
    )
    ap.add_argument(
        "--aureole-to",
        type=float,
        default=None,
        help="eastern end of mapped aureole (m along section)",
    )
    ap.add_argument(
        "--sill-max-thickness", type=float, default=600.0, help="upper bound defining 'tabular' (m)"
    )
    ap.add_argument(
        "--sill-top-max-z",
        type=float,
        default=None,
        help="max z-down of sheet top (m); restricts H1 to geologically "
        "meaningful SHALLOW sheets (e.g. -50 = no deeper than 50 m a.s.l.)",
    )
    ap.add_argument(
        "--min-cover",
        type=float,
        default=100.0,
        help="minimum cover between ground and sheet top (m)",
    )
    ap.add_argument(
        "--h2-top-z",
        type=float,
        default=0.0,
        help="z (down, m) of the trapezoid top; 0 = sea level",
    )
    ap.add_argument("--accept-factor", type=float, default=1.2)
    ap.add_argument("--section-name", default="Section")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy.optimize import least_squares

    df = pd.read_csv(args.profile).sort_values("dist_m")
    dist = df["dist_m"].to_numpy(float)
    obs = df[args.value_col].to_numpy(float)
    if "elev_m" in df.columns:
        elev = df["elev_m"].to_numpy(float)
    else:
        print("WARNING: no elev_m column; using z_obs = 0")
        elev = np.zeros_like(dist)

    excl = np.zeros(len(dist), bool)
    for xd in args.exclude_dist:
        excl |= np.abs(dist - xd) <= args.exclude_tol
    keep = ~excl
    xk, yk, zk = dist[keep], obs[keep], -elev[keep]  # observe at real station heights
    n = int(keep.sum())
    drho = float(args.drho)

    # ---- fits ----
    p2, r2 = fit_h2(xk, yk, zk, drho, args.h2_top_z, least_squares)
    p1, r1 = fit_h1(
        xk,
        yk,
        zk,
        drho,
        dist,
        elev,
        cover=args.min_cover,
        t_max=args.sill_max_thickness,
        aur_from=args.aureole_from,
        aur_to=args.aureole_to,
        top_max_z=args.sill_top_max_z,
        least_squares=least_squares,
    )
    k1 = k2 = 4
    daic = n * np.log((r1**2) / (r2**2)) + 2 * (k1 - k2)  # AIC(H1) - AIC(H2)

    x0, ht, hb, zb = p2
    inset = ht - hb
    dip = np.degrees(np.arctan2(zb - args.h2_top_z, abs(inset))) if abs(inset) > 1 else 90.0
    dip_note = "inward (narrowing down)" if inset > 0 else "outward (widening down)"

    xl, xr, zt, t = p1
    thr = args.accept_factor * r2

    print("== H2 steep/compact trapezoid ==")
    print(f"  top x: {x0 - ht:.0f} - {x0 + ht:.0f} m, base depth: {zb:.0f} m")
    print(f"  wall dip ~ {dip:.0f} deg ({dip_note}); RMS = {r2:.3f} mGal")
    print("== H1 tabular sheet ==")
    print(f"  span: {xl:.0f} - {xr:.0f} m, top z: {zt:.0f} m, thickness: {t:.0f} m")
    print(f"  interior slab amplitude = {slab_mgal(drho, t):.2f} mGal; RMS = {r1:.3f} mGal")
    print(f"== verdict (accept if RMS <= {args.accept_factor:.2f} x {r2:.3f} = {thr:.3f}) ==")
    if r1 <= thr:
        print(f"  H1 NOT excluded by gravity alone (dAIC = {daic:+.1f}).")
        if args.aureole_from is None:
            print("  -> supply --aureole-from/--aureole-to to run the decisive constrained test.")
    else:
        print(f"  H1 EXCLUDED at this level: RMS {r1:.3f} > {thr:.3f} (dAIC = {daic:+.1f}).")

    # ---- thickness scan beneath the aureole (decisive number) ----
    t_admiss = None
    if args.aureole_from is not None and args.aureole_to is not None:
        af, atx = float(args.aureole_from), float(args.aureole_to)
        span = (dist >= af) & (dist <= atx)
        emin = (
            float(elev[span].min())
            if span.any()
            else float(np.interp(0.5 * (af + atx), dist, elev))
        )
        zt_fix = args.min_cover - emin  # shallowest admissible sheet top
        pred_h2 = talwani_gz_polygon(xk, zk, trapezoid(*p2, args.h2_top_z), drho)
        print("== composite scan: best H2 body + sheet pinned beneath the aureole ==")
        print(f"  sheet span fixed {af:.0f}-{atx:.0f} m, top fixed {zt_fix:.0f} m (z-down)")
        ts = np.arange(0.0, args.sill_max_thickness + 1, 25.0)
        rms_ts = []
        for tt in ts:
            add = 0.0 if tt == 0 else talwani_gz_polygon(xk, zk, sheet(af, atx, zt_fix, tt), drho)
            rms_ts.append(rms_of(pred_h2 + add, yk))
        rms_ts = np.asarray(rms_ts)
        ok = rms_ts <= thr
        t_admiss = float(ts[ok].max()) if ok.any() else 0.0
        for tt in (0.0, 200.0, 400.0, 600.0):
            j = int(np.argmin(np.abs(ts - tt)))
            print(f"    t = {ts[j]:4.0f} m -> RMS = {rms_ts[j]:.3f} mGal")
        print(
            f"  max admissible sheet thickness beneath aureole: ~{t_admiss:.0f} m "
            f"(slab amp {slab_mgal(drho, t_admiss):.2f} mGal)"
        )
        print("  (slab interior amplitude is depth-independent -> deepening cannot rescue H1)")

    # ---------------- figure ----------------
    xpad = 0.04 * (dist.max() - dist.min())
    x_lo = min(dist.min(), xl, x0 - ht) - xpad
    x_hi = max(dist.max(), xr, x0 + ht) + xpad
    xd = np.linspace(x_lo, x_hi, 500)
    zd = -np.interp(xd, dist, elev)
    c2 = talwani_gz_polygon(xd, zd, trapezoid(*p2, args.h2_top_z), drho)
    c1 = talwani_gz_polygon(xd, zd, sheet(*p1), drho)

    plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.8})
    fig, (axg, axm) = plt.subplots(
        2, 1, figsize=(7.5, 6.6), sharex=True, gridspec_kw={"height_ratios": [1.0, 1.5]}
    )

    axg.plot(
        xd * KM,
        c2,
        "-",
        color="#c0392b",
        lw=2.0,
        zorder=3,
        label=f"H2 steep compact (RMS {r2:.2f})",
    )
    axg.plot(
        xd * KM,
        c1,
        "--",
        color="#1f4e79",
        lw=2.0,
        zorder=3,
        label=f"H1 tabular sheet (RMS {r1:.2f})",
    )
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
    box = f"$\\Delta$AIC (H1$-$H2) = {daic:+.1f}\naccept: RMS $\\leq$ {thr:.2f} mGal"
    if t_admiss is not None:
        box += f"\nmax sheet under aureole $\\approx$ {t_admiss:.0f} m"
    axg.text(
        0.985,
        0.06,
        box,
        transform=axg.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.92),
    )
    axg.set_title(
        f"{args.section_name}  —  end-member test: tabular vs steep", fontsize=11, loc="left"
    )

    topo_z = -elev * KM
    axm.fill_between(dist * KM, topo_z, 0.0, color="#efe7dd", zorder=0)
    axm.plot(dist * KM, topo_z, color="#6e4b3a", lw=1.4, zorder=6, label="Ground surface")
    v2 = trapezoid(*p2, args.h2_top_z)
    axm.fill(
        v2[:, 0] * KM,
        v2[:, 1] * KM,
        facecolor="#d9544d",
        alpha=0.35,
        edgecolor="#922b21",
        lw=2.0,
        zorder=3,
        label="H2 steep compact",
    )
    v1 = sheet(*p1)
    axm.fill(
        v1[:, 0] * KM,
        v1[:, 1] * KM,
        facecolor="#3b7dd8",
        alpha=0.25,
        edgecolor="#1f4e79",
        lw=2.0,
        hatch="///",
        zorder=4,
        label="H1 tabular sheet",
    )
    axm.axhline(0.0, color="#34607d", lw=0.9, ls="--", zorder=2)
    axm.text(x_lo * KM, 0.0, " sea level", color="#34607d", fontsize=7.5, va="bottom", ha="left")

    if args.aureole_from is not None and args.aureole_to is not None:
        ya = (
            -(float(np.interp(0.5 * (args.aureole_from + args.aureole_to), dist, elev)) * KM) - 0.12
        )
        axm.plot(
            [args.aureole_from * KM, args.aureole_to * KM],
            [ya, ya],
            color="#b9770e",
            lw=3.0,
            solid_capstyle="butt",
            zorder=7,
        )
        axm.text(
            0.5 * (args.aureole_from + args.aureole_to) * KM,
            ya - 0.03,
            "mapped aureole (input)",
            color="#b9770e",
            fontsize=7.5,
            ha="center",
            va="bottom",
        )

    axm.set_ylabel("Depth below sea level (km)")
    axm.set_xlabel("Distance along section (km)")
    axm.grid(alpha=0.2)
    axm.set_xlim(x_lo * KM, x_hi * KM)
    zmax = max(float(v2[:, 1].max()), float(v1[:, 1].max()))
    emax = float(elev.max()) * KM
    axm.set_ylim(-(emax * 1.7 + 0.05), zmax * KM * 1.12)
    axm.invert_yaxis()
    axm.legend(loc="lower left", fontsize=7.5, framealpha=0.92)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300)
    fig.savefig(args.out.with_suffix(".pdf"))
    print(f"Saved: {args.out}")
    print(f"Saved: {args.out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
