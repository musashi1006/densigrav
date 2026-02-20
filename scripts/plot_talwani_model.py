#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import yaml  # PyYAML
except ImportError as e:
    raise SystemExit("PyYAML が必要です: pip install pyyaml") from e


def load_model_yaml(path: Path, y_axis: str = "zdown") -> tuple[float, np.ndarray]:
    """
    Load Talwani polygon model from YAML.

    Expected YAML:
      density_contrast_kgm3: 78.0
      vertices_xz_m:
        - [x1, z1]
        - [x2, z2]
        ...

    y_axis:
      - "zdown": second column is z (depth, positive downward)
      - "elev":  second column is elevation (positive upward). Converted to zdown via z = -elev
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if "density_contrast_kgm3" not in data or "vertices_xz_m" not in data:
        raise ValueError("model.yaml must contain density_contrast_kgm3 and vertices_xz_m")

    drho = float(data["density_contrast_kgm3"])
    verts = np.asarray(data["vertices_xz_m"], dtype=float)
    if verts.ndim != 2 or verts.shape[1] != 2 or verts.shape[0] < 3:
        raise ValueError("vertices_xz_m must be an array of shape (n>=3, 2)")

    x = verts[:, 0]
    y = verts[:, 1]

    if y_axis == "elev":
        z = -y  # elevation -> zdown
    elif y_axis == "zdown":
        z = y
    else:
        raise ValueError("y_axis must be 'zdown' or 'elev'")

    poly = np.column_stack([x, z])

    # close polygon if not closed
    if not np.allclose(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])

    return drho, poly


def load_profile_csv(path: Path):
    """Load profile csv if provided (expects dist_m and optionally residual_mgal / talwani_pred_mgal)."""
    import pandas as pd

    df = pd.read_csv(path)
    if "dist_m" not in df.columns:
        raise ValueError("profile csv must contain dist_m column")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot Talwani 2D polygon model (no QGIS) and optional profile curves."
    )
    ap.add_argument("--model", type=Path, required=True, help="Path to model.yaml")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output image path (png/pdf). If omitted, show window.",
    )
    ap.add_argument(
        "--y-axis",
        choices=["zdown", "elev"],
        default="zdown",
        help="How to interpret the 2nd coordinate in vertices_xz_m (default: zdown).",
    )
    ap.add_argument(
        "--equal-aspect", action="store_true", help="Use equal aspect ratio for the section plot."
    )
    ap.add_argument("--annotate", action="store_true", help="Annotate vertex indices.")
    ap.add_argument(
        "--profile",
        type=Path,
        default=None,
        help="Optional profile csv to plot under the section (dist_m + residual_mgal etc).",
    )
    ap.add_argument(
        "--profile-value",
        type=str,
        default="residual_mgal",
        help="Column to plot as observed curve in profile panel (default: residual_mgal).",
    )

    args = ap.parse_args()

    drho, poly = load_model_yaml(args.model, y_axis=args.y_axis)

    if args.profile is None:
        fig, ax = plt.subplots(figsize=(10, 4))
        axes = [ax]
    else:
        fig, (ax, ax2) = plt.subplots(
            nrows=2, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
        )
        axes = [ax, ax2]

    # --- Section plot ---
    ax = axes[0]
    ax.fill(poly[:, 0], poly[:, 1], alpha=0.25)
    ax.plot(poly[:, 0], poly[:, 1], linewidth=2)
    ax.axhline(0.0, linewidth=1, alpha=0.6)
    ax.grid(True, alpha=0.3)

    # Depth should increase downward visually
    ax.invert_yaxis()

    ax.set_xlabel("Distance x (m)")
    ax.set_ylabel("Depth z (m, z-down)")
    ax.set_title(f"Talwani 2D polygon (Δρ = {drho:g} kg/m³)")

    if args.equal_aspect:
        ax.set_aspect("equal", adjustable="datalim")

    if args.annotate:
        for i, (x, z) in enumerate(poly[:-1]):
            ax.text(x, z, str(i), fontsize=8)

    # --- Optional profile plot ---
    if args.profile is not None:
        df = load_profile_csv(args.profile)
        ax2 = axes[1]

        if args.profile_value not in df.columns:
            raise ValueError(
                f"--profile-value '{args.profile_value}' not found in {args.profile}. "
                f"Available: {list(df.columns)}"
            )

        ax2.plot(
            df["dist_m"].to_numpy(), df[args.profile_value].to_numpy(), marker="o", linewidth=1
        )
        if "talwani_pred_mgal" in df.columns:
            ax2.plot(
                df["dist_m"].to_numpy(), df["talwani_pred_mgal"].to_numpy(), marker="o", linewidth=1
            )

        ax2.axhline(0.0, linewidth=1, alpha=0.6)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylabel("mGal")
        ax2.set_xlabel("Distance x (m)")
        ax2.set_title(
            f"Profile ({args.profile_value}"
            + (" + talwani_pred_mgal" if "talwani_pred_mgal" in df.columns else "")
            + ")"
        )

    fig.tight_layout()

    if args.out is None:
        plt.show()
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=200)
        print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
