#!/usr/bin/env python3
"""Thin wrapper around densigrav.section.ensemble.run_ensemble.

Prefer the installed CLI:  densigrav section ensemble --help
This script lets you run the same ensemble figure without installing the package.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from densigrav.section.ensemble import run_ensemble  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--profile", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--value-col", default="residual_mgal")
    ap.add_argument("--section-name", default="Section")
    ap.add_argument("--exclude-dist", type=float, nargs="*", default=[])
    ap.add_argument("--exclude-tol", type=float, default=2.0)
    ap.add_argument("--sigma", type=float, default=0.8)
    ap.add_argument("--drho-sigma", type=float, default=0.0)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--accept-factor", type=float, default=1.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--use-elev",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate stations at z=-elev_m (default). --no-use-elev uses sea level z=0",
    )
    ap.add_argument("--no-equal-aspect", action="store_true")
    a = ap.parse_args()

    s = run_ensemble(
        a.model,
        a.profile,
        a.out,
        value_col=a.value_col,
        section_name=a.section_name,
        exclude_dist=a.exclude_dist,
        exclude_tol=a.exclude_tol,
        sigma=a.sigma,
        drho_sigma=a.drho_sigma,
        n=a.n,
        accept_factor=a.accept_factor,
        seed=a.seed,
        obs_height="elev" if a.use_elev else "sealevel",
        equal_aspect=not a.no_equal_aspect,
    )
    print(f"Saved: {s['out']}")
    print(f"Saved: {s['pdf']}")
    print(f"Saved ensemble parameters: {s['params_csv']}")
    print(
        f"  draws={s['n_draws']}, accepted={s['n_accepted']} at RMS <= "
        f"{a.accept_factor:.2f} x {s['best_rms_mgal']:.2f} mGal"
    )
    bd, cx = s["base_depth_m"], s["center_x0_m"]
    print(
        f"  best-fit: x0={s['best_center_x0_m']:.0f} m, base depth={s['best_base_depth_m']:.0f} m, "
        f"RMS={s['rms_mgal']:.2f} mGal, VR={s['variance_reduction_pct']:.0f}%"
    )
    print(f"  base depth = {bd[0]:.0f} [{bd[1]:.0f}, {bd[2]:.0f}] m (5-95%)")
    print(f"  center x0  = {cx[0]:.0f} [{cx[1]:.0f}, {cx[2]:.0f}] m (5-95%)")


if __name__ == "__main__":
    main()
