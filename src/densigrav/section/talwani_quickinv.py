from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .talwani2d import talwani_gz_polygon
from .talwani_io import TalwaniModel


@dataclass(frozen=True)
class TrapezoidParams:
    x0_m: float
    z_top_m: float
    z_bottom_m: float
    halfwidth_top_m: float
    halfwidth_bottom_m: float
    density_contrast_kgm3: float


def _params_to_model(p: TrapezoidParams) -> TalwaniModel:
    x0 = p.x0_m
    zt = p.z_top_m
    zb = p.z_bottom_m
    wt = p.halfwidth_top_m
    wb = p.halfwidth_bottom_m
    verts = np.asarray(
        [
            [x0 - wt, zt],
            [x0 + wt, zt],
            [x0 + wb, zb],
            [x0 - wb, zb],
        ],
        dtype=float,
    )
    return TalwaniModel(density_contrast_kgm3=p.density_contrast_kgm3, vertices_xz_m=verts)


def quick_invert_trapezoid(
    x_m: np.ndarray,
    obs_mgal: np.ndarray,
    *,
    z_obs_m: np.ndarray | None = None,
    density_contrast_kgm3: float = 300.0,
    max_nfev: int = 200,
) -> tuple[TalwaniModel, dict[str, float]]:
    """
    Quick inversion for a single trapezoid body.
    Fits: x0, z_top, z_bottom, halfwidth_top, halfwidth_bottom.
    Density contrast fixed (入口体験のため).
    """
    x = np.asarray(x_m, dtype=float).ravel()
    y = np.asarray(obs_mgal, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError("x_m and obs_mgal must have the same shape.")

    if z_obs_m is None:
        z_obs = np.zeros_like(x)
    else:
        z_obs = np.asarray(z_obs_m, dtype=float).ravel()

    try:
        from scipy.optimize import least_squares
    except Exception as e:
        raise RuntimeError(
            'scipy is required for quick inversion. Install extras: ".[grid]"'
        ) from e

    # Initial guess: centered, shallow top, moderate thickness
    x0_0 = float(x[np.argmax(np.abs(y))])
    zt_0 = 200.0
    zb_0 = 1200.0
    wt_0 = 300.0
    wb_0 = 600.0

    p0 = np.array([x0_0, zt_0, zb_0, wt_0, wb_0], dtype=float)

    # bounds (z positive down)
    xmin, xmax = float(np.min(x)), float(np.max(x))
    bounds_lo = np.array([xmin, 10.0, 50.0, 10.0, 10.0], dtype=float)
    bounds_hi = np.array([xmax, 5000.0, 10000.0, 10000.0, 10000.0], dtype=float)

    def residual(pvec: np.ndarray) -> np.ndarray:
        x0, zt, zb, wt, wb = [float(v) for v in pvec]
        if zb <= zt:
            return 1e6 * np.ones_like(y)
        model = _params_to_model(
            TrapezoidParams(
                x0_m=x0,
                z_top_m=zt,
                z_bottom_m=zb,
                halfwidth_top_m=wt,
                halfwidth_bottom_m=wb,
                density_contrast_kgm3=float(density_contrast_kgm3),
            )
        )
        pred = talwani_gz_polygon(x, z_obs, model.vertices_xz_m, model.density_contrast_kgm3)
        return pred - y

    res = least_squares(
        residual,
        p0,
        bounds=(bounds_lo, bounds_hi),
        max_nfev=max_nfev,
    )

    x0, zt, zb, wt, wb = [float(v) for v in res.x]
    model = _params_to_model(
        TrapezoidParams(
            x0_m=x0,
            z_top_m=zt,
            z_bottom_m=zb,
            halfwidth_top_m=wt,
            halfwidth_bottom_m=wb,
            density_contrast_kgm3=float(density_contrast_kgm3),
        )
    )

    stats = {
        "rmse_mgal": float(np.sqrt(np.mean(res.fun**2))),
        "nfev": float(res.nfev),
        "x0_m": x0,
        "z_top_m": zt,
        "z_bottom_m": zb,
        "halfwidth_top_m": wt,
        "halfwidth_bottom_m": wb,
        "density_contrast_kgm3": float(density_contrast_kgm3),
    }
    return model, stats
