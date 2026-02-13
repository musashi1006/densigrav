from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


class GriddingError(Exception):
    pass


@dataclass(frozen=True)
class GridResult:
    grid: np.ndarray  # (ny, nx)
    transform: object  # rasterio Affine
    crs_wkt: str  # e.g. "EPSG:6677"
    resolution_m: float
    bounds: Tuple[float, float, float, float]  # west, south, east, north


def _require_rasterio() -> None:
    try:
        import rasterio  # noqa: F401
    except Exception as e:
        raise GriddingError("rasterio is required. Install with: pip install -e '.[raster]'") from e


def _require_verde() -> None:
    try:
        import verde  # noqa: F401
    except Exception as e:
        raise GriddingError("verde is required. Install with: pip install -e '.[grid]'") from e


def _grid_axis(
    west: float, east: float, north: float, south: float, res: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return x centers increasing, y centers decreasing (north-up raster).
    """
    if east <= west or north <= south:
        raise GriddingError("Invalid bounds for gridding.")

    nx = int(np.ceil((east - west) / res))
    ny = int(np.ceil((north - south) / res))
    nx = max(nx, 1)
    ny = max(ny, 1)

    x = west + res * (0.5 + np.arange(nx))
    y = north - res * (0.5 + np.arange(ny))  # descending
    return x, y


def _default_region_from_points(
    x: np.ndarray,
    y: np.ndarray,
    *,
    padding_m: float,
) -> Tuple[float, float, float, float]:
    xmin = float(np.nanmin(x))
    xmax = float(np.nanmax(x))
    ymin = float(np.nanmin(y))
    ymax = float(np.nanmax(y))
    return (xmin - padding_m, ymin - padding_m, xmax + padding_m, ymax + padding_m)


def grid_from_points(
    *,
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    crs_wkt: str,
    resolution_m: float,
    region: Optional[Tuple[float, float, float, float]] = None,  # west,south,east,north
) -> GridResult:
    """
    Grid scattered points using Verde. Output is a north-up grid suitable for GeoTIFF.

    v0.1 strategy:
      - BlockReduce (mean) with spacing=resolution_m
      - If enough points: Spline
        else: KNeighbors
    """
    _require_rasterio()
    _require_verde()

    import verde as vd
    from rasterio.transform import from_origin

    res = float(resolution_m)
    if res <= 0:
        raise GriddingError("resolution_m must be > 0")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    v = np.asarray(values, dtype=float)

    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
    if m.sum() < 3:
        raise GriddingError("Not enough valid points to grid (need >= 3).")

    xr = x[m]
    yr = y[m]
    vr = v[m]

    # region
    if region is None:
        pad = res * 2.0
        west, south, east, north = _default_region_from_points(xr, yr, padding_m=pad)
    else:
        west, south, east, north = region

    # reduce
    reducer = vd.BlockReduce(reduction="mean", spacing=res)
    coords_red, data_red = reducer.filter((xr, yr), vr)

    # choose estimator
    n_red = int(np.size(data_red))
    if n_red >= 20:
        estimator = vd.Spline(damping=1e-10)
    else:
        k = max(3, min(10, n_red))
        estimator = vd.KNeighbors(k=k)

    estimator.fit(coords_red, data_red)

    # grid axis (north-up)
    xg, yg = _grid_axis(west, east, north, south, res)
    xx, yy = np.meshgrid(xg, yg)
    pred = estimator.predict((xx.ravel(), yy.ravel())).reshape(xx.shape)

    # raster transform: top-left corner of pixel (west, north)
    transform = from_origin(west, north, res, res)

    return GridResult(
        grid=pred.astype(float),
        transform=transform,
        crs_wkt=str(crs_wkt),
        resolution_m=res,
        bounds=(west, south, east, north),
    )


def write_geotiff(
    *,
    out_path: Path,
    grid: np.ndarray,
    transform: object,
    crs_wkt: str,
    overwrite: bool = False,
    nodata: float = -9999.0,
) -> Path:
    _require_rasterio()
    import rasterio

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if overwrite and out_path.exists():
        out_path.unlink()

    arr = np.asarray(grid, dtype=np.float32)
    arr_out = np.where(np.isfinite(arr), arr, np.float32(nodata))

    height, width = arr_out.shape
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": crs_wkt,
        "transform": transform,
        "nodata": float(nodata),
        "compress": "deflate",
        "tiled": True,
    }

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr_out, 1)

    return out_path
