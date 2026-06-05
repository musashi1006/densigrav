#!/usr/bin/env python3
"""Thin wrapper around densigrav.section.plotting.plot_section_model.

Prefer the installed CLI:  densigrav section plot --help
This script lets you run the same figure without installing the package.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from densigrav.section.plotting import plot_section_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--profile", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--value-col", default="residual_mgal")
    ap.add_argument("--section-name", default="Section")
    ap.add_argument("--exclude-dist", type=float, nargs="*", default=[])
    ap.add_argument("--exclude-tol", type=float, default=2.0)
    ap.add_argument("--obs-height", choices=["sealevel", "elev"], default="sealevel")
    ap.add_argument("--equal-aspect", action="store_true")
    a = ap.parse_args()

    s = plot_section_model(
        a.model,
        a.profile,
        a.out,
        value_col=a.value_col,
        section_name=a.section_name,
        exclude_dist=a.exclude_dist,
        exclude_tol=a.exclude_tol,
        obs_height=a.obs_height,
        equal_aspect=a.equal_aspect,
    )
    print(f"Saved: {s['out']}")
    print(f"Saved: {s['pdf']}")
    print(
        f"  RMS misfit = {s['rms_mgal']:.3f} mGal | variance reduction = "
        f"{s['variance_reduction_pct']:.1f}% | N = {s['n']}"
    )


if __name__ == "__main__":
    main()
