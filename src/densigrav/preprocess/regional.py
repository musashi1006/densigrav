from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class RegionalError(Exception):
    pass


@dataclass(frozen=True)
class RegionalResult:
    regional: np.ndarray
    residual: np.ndarray


def fit_polynomial_trend(
    x: np.ndarray,
    y: np.ndarray,
    data: np.ndarray,
    *,
    order: int = 2,
) -> RegionalResult:
    """
    Fit a polynomial trend (Verde Trend) and return regional + residual.
    x, y in meters (projected CRS).
    """
    try:
        import verde as vd
    except Exception as e:
        raise RegionalError("verde is required. Install with: pip install -e '.[grid]'") from e

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d = np.asarray(data, dtype=float)

    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(d)
    if m.sum() < max(3, (order + 1) * 2):
        raise RegionalError(f"Not enough valid points for polynomial trend (order={order}).")

    coords = (x[m], y[m])
    trend = vd.Trend(degree=int(order))
    trend.fit(coords, d[m])

    regional = np.full_like(d, np.nan, dtype=float)
    regional[m] = trend.predict(coords)

    residual = d - regional
    return RegionalResult(regional=regional, residual=residual)
