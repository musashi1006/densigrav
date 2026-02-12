from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import rasterio


class DemIOError(Exception):
    pass


@dataclass(frozen=True)
class DemPrepareStats:
    n_points: int
    n_missing_before: int
    n_filled_from_dem: int
    n_missing_after: int
    n_outside: int
    dem_crs: str
    points_crs: str


def _require_geopandas():
    try:
        import geopandas as gpd  # noqa: F401
    except Exception as e:
        raise DemIOError("geopandas is required. Install with: pip install -e '.[geo]'") from e


def _require_rasterio():
    try:
        import rasterio  # noqa: F401
    except Exception as e:
        raise DemIOError("rasterio is required. Install with: pip install -e '.[raster]'") from e


def aoi_from_points_bbox(points_gdf, buffer_m: float):
    """
    AOI polygon = points bbox expanded by buffer (units are CRS units).
    """
    _require_geopandas()
    from shapely.geometry import box

    if points_gdf.empty:
        raise DemIOError("No points provided to build AOI.")

    xmin, ymin, xmax, ymax = points_gdf.total_bounds
    geom = box(xmin, ymin, xmax, ymax)
    if buffer_m and buffer_m > 0:
        geom = geom.buffer(float(buffer_m))
    return geom


def _reproject_shapely_geom(geom, src_crs: str, dst_crs: str):
    """
    Reproject shapely geometry from src_crs to dst_crs.
    """
    if src_crs == dst_crs:
        return geom

    from pyproj import Transformer
    from shapely.ops import transform as shapely_transform

    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

    def _f(x, y, z=None):
        return transformer.transform(x, y)

    return shapely_transform(_f, geom)


def clip_dem_to_geom(dem_path: Path, geom, out_path: Path, overwrite: bool = False) -> Path:
    """
    Clip DEM to polygon geometry. Geometry must be in DEM CRS.
    """
    _require_rasterio()
    from rasterio.mask import mask
    from shapely.geometry import mapping

    dem_path = Path(dem_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if overwrite and out_path.exists():
        out_path.unlink()

    with rasterio.open(dem_path) as src:
        out_image, out_transform = mask(src, [mapping(geom)], crop=True)
        out_meta = src.meta.copy()
        out_meta.update(
            {
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
            }
        )

        with rasterio.open(out_path, "w", **out_meta) as dst:
            dst.write(out_image)

    return out_path


def sample_dem_to_points(
    dem_path: Path,
    points_gdf,
    *,
    z_col: str = "z",
    fill_only_missing: bool = True,
    error_if_outside: bool = False,
) -> Tuple[object, DemPrepareStats]:
    """
    Sample DEM band1 to point locations and fill z_col.
    Points are reprojected to DEM CRS for sampling if needed.
    Returns (points_gdf_with_z, stats).
    """
    _require_geopandas()
    _require_rasterio()

    dem_path = Path(dem_path)

    if points_gdf.crs is None:
        raise DemIOError("Points GeoDataFrame has no CRS. Set CRS before sampling.")

    with rasterio.open(dem_path) as src:
        dem_crs = src.crs.to_string() if src.crs else ""
        if not dem_crs:
            raise DemIOError(f"DEM has no CRS: {dem_path}")

        # Reproject points to DEM CRS if needed
        pts = points_gdf
        points_crs = pts.crs.to_string()
        if points_crs != dem_crs:
            pts = pts.to_crs(dem_crs)

        xs = np.asarray(pts.geometry.x, dtype=float)
        ys = np.asarray(pts.geometry.y, dtype=float)

        # bounds check
        b = src.bounds
        inside = (xs >= b.left) & (xs <= b.right) & (ys >= b.bottom) & (ys <= b.top)

        # sample only inside
        z = np.full(xs.shape[0], np.nan, dtype=float)
        if inside.any():
            coords = list(zip(xs[inside], ys[inside]))
            vals = np.array([v[0] for v in src.sample(coords)], dtype=float)

            # nodata -> nan
            if src.nodata is not None:
                vals = np.where(vals == float(src.nodata), np.nan, vals)

            z[inside] = vals

        n_outside = int((~inside).sum())
        if error_if_outside and n_outside > 0:
            raise DemIOError(f"{n_outside} points are outside DEM bounds: {dem_path}")

        # Fill into original CRS gdf (keep original CRS for outputs)
        out = points_gdf.copy()

        if z_col not in out.columns:
            out[z_col] = np.nan

        target_before = out[z_col].astype(float).to_numpy()
        missing_before = ~np.isfinite(target_before)
        n_missing_before = int(missing_before.sum())

        target_after = target_before.copy()

        if fill_only_missing:
            if missing_before.any():
                # only fill missing entries with DEM samples
                target_after[missing_before] = z[missing_before]
        else:
            # overwrite all with DEM samples (may include NaN if DEM nodata)
            target_after[:] = z

        missing_after = ~np.isfinite(target_after)
        n_missing_after = int(missing_after.sum())

        # How many were actually filled by DEM (only meaningful in fill_only_missing mode)
        if fill_only_missing:
            filled_from_dem_mask = missing_before & np.isfinite(target_after)
            n_filled_from_dem = int(filled_from_dem_mask.sum())
        else:
            n_filled_from_dem = int(np.isfinite(target_after).sum())

        out[z_col] = target_after

        stats = DemPrepareStats(
            n_points=int(len(out)),
            n_missing_before=n_missing_before,
            n_filled_from_dem=n_filled_from_dem,
            n_missing_after=n_missing_after,
            n_outside=n_outside,
            dem_crs=dem_crs,
            points_crs=points_crs,
        )

        return out, stats

        """
        target = out[z_col].astype(float).to_numpy()
        if fill_only_missing:
            missing = ~np.isfinite(target)
            if missing.any():
                target[missing] = z[missing]
            else:
                target[:] = z
                
        out[z_col] = target

        n_filled = int(np.isfinite(out[z_col].to_numpy()).sum())

        stats = DemPrepareStats(
            n_points=int(len(out)),
            n_filled=n_filled,
            n_outside=n_outside,
            dem_crs=dem_crs,
            points_crs=points_crs,
        )

        return out, stats
        """


def write_dem_prepare_points_gpkg(
    points_gdf, out_path: Path, *, layer: str = "gravity_points", overwrite: bool = False
):
    """
    Write points (with z) to GeoPackage.
    """
    from .gravity_points import write_gpkg

    write_gpkg(points_gdf, out_path, layer=layer, overwrite=overwrite)
