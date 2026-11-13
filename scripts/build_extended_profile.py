#!/usr/bin/env python3
"""Rebuild the eastward-extended Shinshiro profile and its regional-trend variants.

This script is a transcription of the ad-hoc processing run on 2026-07-10 (the
original code was not saved). It was reverse-engineered from the surviving input
and output files and reproduces the five profile CSVs to floating-point precision
(see --verify-against).

Pipeline
--------
1. Project the Nagoya gravity-database points (lon/lat) onto the section line.
   The line is the least-squares regression line through the Otani (2005) station
   positions in a local equirectangular frame; t = along-line distance from the
   westernmost Otani station, d = signed offset (north positive).
2. Estimate the shift between the database along-line distance t and the legacy
   profile distance (10 m grid scan minimising the mismatch std) -> 1130 m.
3. Tie the database Bouguer values to the legacy profile (median difference inside
   the profile range within the tie corridor) -> +0.0638 mGal.
4. Append the database points east of the legacy profile within the extension
   corridor (points co-located with an Otani station are dropped as duplicates).
5. Regional trends: 2-D polynomials (order 0/1/2) fitted to all database points,
   either unmasked or with anomaly masking (MAD-based iterative clipping). The
   trend is evaluated on the section line (d = 0) and subtracted.

Known limitations of the original processing (kept as-is for reproducibility;
see the printed report):
- A single distance shift is only an approximation: the legacy profile distance is
  not a rigid shift of t (per-station shifts range from about 0.7 to 2.2 km).
- The clipping loop runs a fixed number of fits and does not converge for order 2
  (the mask oscillates between 265 and 266 points).
- The tie corridor (1500 m) differs from the extension corridor (1000 m), and the
  regional tie constant (0.064) is a rounded value of the Bouguer tie (0.0638).

Inputs (not distributed with the repository)
--------------------------------------------
--db-csv           gravity-database points; columns: station_id, lon, lat, elev_m,
                   bouguer_mgal
--otani-positions  positions of the thesis stations; columns: lon, lat

Example
-------
python scripts/build_extended_profile.py \
    --db-csv path/to/gravity_db_points.csv \
    --otani-positions path/to/thesis_station_positions.csv \
    --base-profile results/talwani/profile.csv \
    --base-residual results/talwani/profile_residual_v1.csv \
    --out-dir results/2d/section_Shinshiro_repro \
    --verify-against results/2d/section_Shinshiro
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

M_PER_DEG = 111320.0  # metres per degree of latitude used by the original run
OUT_COLUMNS = [
    "dist_m",
    "elev_m",
    "bouguer_mgal",
    "residual_mgal",
    "source",
    "station_id",
    "offline_m",
]


# --------------------------------------------------------------------------- #
# 1. projection onto the section line
# --------------------------------------------------------------------------- #
def section_line(otani: pd.DataFrame) -> dict:
    """Regression line through the Otani stations in a local equirectangular frame."""
    lat0 = float(otani["lat"].mean())
    lon0 = float(otani["lon"].mean())
    kx = M_PER_DEG * np.cos(np.radians(lat0))
    x = (otani["lon"].to_numpy() - lon0) * kx
    y = (otani["lat"].to_numpy() - lat0) * M_PER_DEG
    slope, intercept = np.polyfit(x, y, 1)
    theta = float(np.arctan(slope))
    t = x * np.cos(theta) + (y - intercept) * np.sin(theta)
    return {
        "lon0": lon0,
        "lat0": lat0,
        "kx": kx,
        "theta": theta,
        "intercept": float(intercept),
        "t_min": float(t.min()),
    }


def project(lon: np.ndarray, lat: np.ndarray, line: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (t_along_m, d_offset_m) of lon/lat points relative to the section line."""
    x = (np.asarray(lon, float) - line["lon0"]) * line["kx"]
    y = (np.asarray(lat, float) - line["lat0"]) * M_PER_DEG - line["intercept"]
    c, s = np.cos(line["theta"]), np.sin(line["theta"])
    return x * c + y * s - line["t_min"], -x * s + y * c


# --------------------------------------------------------------------------- #
# 2-3. distance shift and Bouguer tie
# --------------------------------------------------------------------------- #
def _tie_differences(
    db: pd.DataFrame, base: pd.DataFrame, shift_m: float, corridor_m: float
) -> np.ndarray:
    x = db["t_along_m"].to_numpy() + shift_m
    inside = (
        (np.abs(db["d_offset_m"].to_numpy()) <= corridor_m)
        & (x >= base["dist_m"].min())
        & (x <= base["dist_m"].max())
    )
    interp = np.interp(x[inside], base["dist_m"].to_numpy(), base["bouguer_mgal"].to_numpy())
    return interp - db["bouguer_mgal"].to_numpy()[inside]


def estimate_distance_shift(
    db: pd.DataFrame, base: pd.DataFrame, corridor_m: float, step_m: float = 10.0
) -> tuple[float, pd.DataFrame]:
    """Grid-scan the shift (profile distance = t + shift) minimising the mismatch std."""
    rows = []
    for shift in np.arange(0.0, 3000.0 + step_m, step_m):
        diff = _tie_differences(db, base, shift, corridor_m)
        rows.append((shift, len(diff), float(diff.std())))
    scan = pd.DataFrame(rows, columns=["shift_m", "n", "std_mgal"])
    return float(scan.loc[scan["std_mgal"].idxmin(), "shift_m"]), scan


# --------------------------------------------------------------------------- #
# 5. regional trend
# --------------------------------------------------------------------------- #
def design(t_km: np.ndarray, d_km: np.ndarray, order: int) -> np.ndarray:
    cols = [np.ones_like(t_km)]
    if order >= 1:
        cols += [t_km, d_km]
    if order >= 2:
        cols += [t_km * t_km, t_km * d_km, d_km * d_km]
    return np.column_stack(cols)


def fit_poly(t_km, d_km, g, order, mask=None) -> np.ndarray:
    if mask is None:
        mask = np.ones(len(g), bool)
    return np.linalg.lstsq(design(t_km[mask], d_km[mask], order), g[mask], rcond=None)[0]


def masked_fit(t_km, d_km, g, order, n_fits=5, k_sigma=2.5, sigma_floor=0.5):
    """Anomaly-masked polynomial fit by MAD-based iterative clipping.

    Each pass fits the currently kept points, then keeps the points whose residual
    lies within k_sigma * max(1.4826 * MAD, sigma_floor) of the kept-point median.
    The original run used exactly ``n_fits`` = 5 fits without a convergence test.

    Returns (coefficients, mask used for the final fit, history rows).
    """
    mask = np.ones(len(g), bool)
    history = []
    coef = None
    for i in range(n_fits):
        coef = fit_poly(t_km, d_km, g, order, mask)
        res = g - design(t_km, d_km, order) @ coef
        med = np.median(res[mask])
        sigma = max(1.4826 * np.median(np.abs(res[mask] - med)), sigma_floor)
        history.append(
            {
                "fit": i + 1,
                "n_kept": int(mask.sum()),
                "rms_mgal": float(np.sqrt(np.mean(res[mask] ** 2))),
                "sigma_mgal": float(sigma),
            }
        )
        fit_mask = mask
        mask = np.abs(res - med) <= k_sigma * sigma
    return coef, fit_mask, history


def bic(t_km, d_km, g, order, mask) -> tuple[float, float]:
    """Gaussian BIC (n ln(RSS/n) + k ln n) and RMS of an order-`order` fit on `mask`."""
    coef = fit_poly(t_km, d_km, g, order, mask)
    res = (g - design(t_km, d_km, order) @ coef)[mask]
    n, k = len(res), len(coef)
    rss = float(np.sum(res**2))
    return n * np.log(rss / n) + k * np.log(n), float(np.sqrt(rss / n))


def trend_on_line(coef, order, dist_m, shift_m) -> np.ndarray:
    t_km = (np.asarray(dist_m, float) - shift_m) / 1000.0
    return design(t_km, np.zeros_like(t_km), order) @ coef


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def build(a: argparse.Namespace) -> dict:
    db = pd.read_csv(a.db_csv)
    otani = pd.read_csv(a.otani_positions)
    base = pd.read_csv(a.base_profile)
    base_res = pd.read_csv(a.base_residual)
    log = print

    # 1. projection
    line = section_line(otani)
    db["t_along_m"], db["d_offset_m"] = project(db["lon"], db["lat"], line)
    ot_t, ot_d = project(otani["lon"], otani["lat"], line)
    log(f"[1] section line: azimuth N{90 - np.degrees(line['theta']):.2f}E, {len(db)} DB points")

    # 2. distance shift
    if a.dist_shift is None:
        shift, scan = estimate_distance_shift(db, base, a.extension_corridor)
        log(f"[2] distance shift (scan, corridor {a.extension_corridor:.0f} m): {shift:.0f} m")
    else:
        shift = a.dist_shift
        log(f"[2] distance shift (given): {shift:.0f} m")

    # 3. Bouguer tie
    diff = _tie_differences(db, base, shift, a.tie_corridor)
    tie = float(np.median(diff))
    log(
        f"[3] Bouguer tie (profile - DB), corridor {a.tie_corridor:.0f} m: median {tie:+.4f} mGal, "
        f"std {diff.std():.2f}, n={len(diff)}"
    )

    # 4. extension points
    x = db["t_along_m"].to_numpy() + shift
    cand = (np.abs(db["d_offset_m"].to_numpy()) <= a.extension_corridor) & (
        x > base["dist_m"].max()
    )
    # drop DB points co-located with an Otani station (the same site, already in the profile)
    near_otani = np.array(
        [
            np.hypot(ot_t - t, ot_d - d).min() <= a.duplicate_tol
            for t, d in zip(db["t_along_m"], db["d_offset_m"])
        ]
    )
    sel = cand & ~near_otani
    dropped = db.loc[cand & near_otani, "station_id"].tolist()
    log(f"[4] extension points: {int(sel.sum())} kept, dropped as Otani duplicates: {dropped}")

    ext = db.loc[sel].assign(dist_m=x[sel]).sort_values("dist_m")
    part_otani = pd.DataFrame(
        {
            "dist_m": base["dist_m"],
            "elev_m": base["elev_m"],
            "bouguer_mgal": base["bouguer_mgal"],
            "source": "otani2005",
            "station_id": np.nan,
            "offline_m": 0.0,
        }
    )
    part_db = pd.DataFrame(
        {
            "dist_m": ext["dist_m"].to_numpy(),
            "elev_m": ext["elev_m"].to_numpy(),
            "bouguer_mgal": ext["bouguer_mgal"].to_numpy() + tie,
            "source": "nagoyaDB",
            "station_id": ext["station_id"].to_numpy(),
            "offline_m": np.round(ext["d_offset_m"].to_numpy()),
        }
    )
    prof = pd.concat([part_otani, part_db], ignore_index=True)
    n_ot = len(part_otani)

    # 5. regional trends
    t_km = db["t_along_m"].to_numpy() / 1000.0
    d_km = db["d_offset_m"].to_numpy() / 1000.0
    g = db["bouguer_mgal"].to_numpy()
    dist = prof["dist_m"].to_numpy()
    boug = prof["bouguer_mgal"].to_numpy()

    # 5a. constant regional inherited from the legacy residual profile
    const = round(float(np.median(base_res["bouguer_mgal"] - base_res["residual_mgal"])), 6)
    res_const = boug - const
    res_const[:n_ot] = base_res["residual_mgal"].to_numpy()
    log(f"[5a] constant regional (legacy): {const:.3f} mGal")

    outputs = {
        "profile_extended_east_v4.csv": res_const,
        "profile_extended_east_v4_constreg.csv": res_const,
    }

    # 5b. unmasked order-2 trend
    coef = fit_poly(t_km, d_km, g, 2)
    outputs["profile_extended_east_v4_mapreg.csv"] = boug - (
        trend_on_line(coef, 2, dist, shift) + a.regional_tie
    )

    # 5c. anomaly-masked trends
    masks = {}
    for order in (1, 2):
        coef, mask, hist = masked_fit(t_km, d_km, g, order, a.n_fits, a.clip_sigma, a.sigma_floor)
        masks[order] = mask
        trend = trend_on_line(coef, order, dist, shift) + a.regional_tie
        outputs[f"profile_extended_east_v5_maskedreg{order}.csv"] = boug - trend
        steps = ", ".join(f"{h['n_kept']}pts/{h['rms_mgal']:.3f}" for h in hist)
        log(
            f"[5c] masked order-{order}: final fit on {int(mask.sum())} points "
            f"({int((~mask).sum())} excluded); per-fit kept/RMS: {steps}"
        )
        log(
            f"     trend along line: {trend[0]:.2f} mGal at {dist[0]:.0f} m -> "
            f"{trend[-1]:.2f} mGal at {dist[-1]:.0f} m"
        )

    # model comparison on one common background mask (order-2 mask)
    common = masks[2]
    log(f"[5d] order comparison on the common order-2 mask (n={int(common.sum())}):")
    for order in (0, 1, 2):
        b, rms = bic(t_km, d_km, g, order, common)
        log(f"     order {order}: RMS {rms:.3f} mGal, BIC {b:.1f}")

    a.out_dir.mkdir(parents=True, exist_ok=True)
    for name, residual in outputs.items():
        out = prof.assign(residual_mgal=residual)[OUT_COLUMNS]
        out.to_csv(a.out_dir / name, index=False)
    mask_df = db[["station_id", "lon", "lat", "t_along_m", "d_offset_m", "bouguer_mgal"]].assign(
        background_order1=masks[1], background_order2=masks[2]
    )
    mask_df.to_csv(a.out_dir / "regional_background_mask.csv", index=False)
    log(f"wrote {len(outputs)} profiles + regional_background_mask.csv to {a.out_dir}")
    return outputs


def verify(out_dir: Path, ref_dir: Path, names) -> bool:
    ok = True
    print(f"\nverification against {ref_dir}:")
    for name in names:
        ref_path = ref_dir / name
        if not ref_path.exists():
            print(f"  {name}: reference missing")
            ok = False
            continue
        new, ref = pd.read_csv(out_dir / name), pd.read_csv(ref_path)
        same_shape = new.shape == ref.shape and list(new.columns) == list(ref.columns)
        if not same_shape:
            print(f"  {name}: shape/columns differ {new.shape} vs {ref.shape}")
            ok = False
            continue
        num = ["dist_m", "elev_m", "bouguer_mgal", "residual_mgal", "offline_m"]
        max_abs = float(np.abs(new[num].to_numpy() - ref[num].to_numpy()).max())
        text_same = (
            new[["source", "station_id"]]
            .fillna("")
            .equals(ref[["source", "station_id"]].fillna(""))
        )
        bytes_same = (out_dir / name).read_bytes().replace(b"\r\n", b"\n") == (
            ref_path.read_bytes().replace(b"\r\n", b"\n")
        )
        passed = text_same and max_abs < 1e-9
        ok &= passed
        print(
            f"  {name}: {'OK' if passed else 'MISMATCH'} "
            f"(max |diff| = {max_abs:.1e}, text identical: {bytes_same})"
        )
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--db-csv", required=True, type=Path, help="Nagoya DB points (lon, lat, ...)")
    ap.add_argument("--otani-positions", required=True, type=Path, help="Otani station lon/lat")
    ap.add_argument("--base-profile", required=True, type=Path, help="legacy profile (19 points)")
    ap.add_argument(
        "--base-residual", required=True, type=Path, help="legacy profile with residual_mgal"
    )
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--verify-against", type=Path, default=None)
    ap.add_argument("--extension-corridor", type=float, default=1000.0, help="m (default 1000)")
    ap.add_argument("--tie-corridor", type=float, default=1500.0, help="m (default 1500)")
    ap.add_argument("--duplicate-tol", type=float, default=1.0, help="m (default 1)")
    ap.add_argument(
        "--dist-shift", type=float, default=None, help="m; default: estimated by grid scan"
    )
    ap.add_argument("--regional-tie", type=float, default=0.064, help="mGal added to DB trends")
    ap.add_argument("--n-fits", type=int, default=5, help="clipping fits (original run: 5)")
    ap.add_argument("--clip-sigma", type=float, default=2.5)
    ap.add_argument("--sigma-floor", type=float, default=0.5, help="mGal")
    a = ap.parse_args()

    outputs = build(a)
    if a.verify_against is not None and not verify(a.out_dir, a.verify_against, outputs):
        sys.exit(1)


if __name__ == "__main__":
    main()
