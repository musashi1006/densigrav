from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


class PreprocessError(Exception):
    pass


@dataclass(frozen=True)
class Step1Stats:
    n_points: int
    z_missing_before: int
    z_filled_from_dem: int
    z_missing_after: int
    out_points: Path
    out_dem_clipped: Path
    bouguer_density_kgm3: float
    terrain_density_kgm3: float
    outer_radius_m: float


def _require(mod: str, hint: str) -> None:
    try:
        __import__(mod)
    except Exception as e:
        raise PreprocessError(f"Missing dependency: {mod}. {hint}") from e


def assert_projected(crs_str: str) -> None:
    _require("pyproj", "Install with: pip install -e '.[geo]'")
    from pyproj import CRS

    crs = CRS.from_user_input(crs_str)
    if crs.is_geographic:
        raise PreprocessError(
            "v0.1 requires a projected CRS (meters). "
            f"Got geographic CRS: {crs_str}. Reproject points/DEM and set project.crs."
        )


def bouguer_slab_mgal(height_m: np.ndarray, density_kgm3: float) -> np.ndarray:
    """
    Simple Bouguer slab attraction in mGal:
      g = 2*pi*G*rho*h  [m/s^2] -> *1e5 [mGal]
    """
    g = 6.67430e-11
    return (2.0 * np.pi * g * float(density_kgm3) * height_m) * 1e5


def _read_dem_grid(dem_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, float, float]:
    """
    Return: topo(height x width), easting(width), northing(height), crs_str, dx, dy
    northing/easting are pixel centers in DEM CRS.
    """
    _require("rasterio", "Install with: pip install -e '.[raster]'")
    import rasterio

    with rasterio.open(dem_path) as src:
        crs = src.crs.to_string() if src.crs else ""
        if not crs:
            raise PreprocessError(f"DEM has no CRS: {dem_path}")

        topo = src.read(1, masked=True).filled(np.nan).astype(float)
        t = src.transform

        dx = float(t.a)
        dy = float(t.e)  # usually negative
        x0 = float(t.c)
        y0 = float(t.f)

        easting = x0 + (np.arange(src.width) + 0.5) * dx
        northing = y0 + (np.arange(src.height) + 0.5) * dy

    # Make axis increasing for stability: if decreasing, flip arrays
    if easting.size >= 2 and easting[1] < easting[0]:
        easting = easting[::-1]
        topo = topo[:, ::-1]
        dx = -dx
    if northing.size >= 2 and northing[1] < northing[0]:
        northing = northing[::-1]
        topo = topo[::-1, :]
        dy = -dy

    return topo, easting, northing, crs, float(abs(dx)), float(abs(dy))


def dem_to_prisms(
    dem_path: Path,
    *,
    density_kgm3: float,
    reference_m: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Convert DEM to rectangular prisms (west,east,south,north,bottom,top) and density array.
    Only uses cells where top > reference and finite.
    """
    topo, easting, northing, crs, dx, dy = _read_dem_grid(dem_path)

    # Surface relative to reference (sea level). Negative -> reference.
    top = np.where(np.isfinite(topo), topo, reference_m)
    top = np.maximum(top, reference_m)

    # Build prism bounds for each cell
    xx, yy = np.meshgrid(easting, northing)

    west = xx - dx / 2.0
    east = xx + dx / 2.0
    south = yy - dy / 2.0
    north = yy + dy / 2.0

    bottom = np.full_like(top, float(reference_m), dtype=float)

    mask = np.isfinite(top) & (top > reference_m)

    prisms = np.stack([west, east, south, north, bottom, top], axis=-1).reshape(-1, 6)
    dens = np.full(prisms.shape[0], float(density_kgm3), dtype=float)

    mask_flat = mask.reshape(-1)
    prisms = prisms[mask_flat]
    dens = dens[mask_flat]

    return prisms, dens, crs


def compute_tc_cba_from_ba(
    *,
    points_gdf,
    dem_path: Path,
    out_dem_clipped: Path,
    out_points: Path,
    bouguer_density_kgm3: float,
    terrain_density_kgm3: float,
    outer_radius_m: float,
    station_epsilon_m: float,
    overwrite: bool,
    layer: str = "gravity_points",
) -> Step1Stats:
    """
    Input points_gdf must have:
      - geometry (Point)
      - g_in (Bouguer anomaly in mGal)  [created by standardize_points(target_unit="mGal")]
      - z (elevation in m) optional (NaN will be filled from DEM)

    Output gpkg adds:
      - slab_mgal
      - terrain_effect_mgal
      - tc_mgal
      - cba_mgal
    """
    _require("geopandas", "Install with: pip install -e '.[geo]'")
    _require("harmonica", "Install with: pip install -e '.[grav]'")
    _require("rasterio", "Install with: pip install -e '.[raster]'")

    import harmonica as hm
    import rasterio

    from densigrav.io.dem import (
        _reproject_shapely_geom,
        aoi_from_points_bbox,
        clip_dem_to_geom,
        sample_dem_to_points,
    )
    from densigrav.io.gravity_points import write_gpkg

    if points_gdf.crs is None:
        raise PreprocessError("Points GeoDataFrame CRS is missing.")

    if "g_in" not in points_gdf.columns:
        raise PreprocessError(
            "Missing required column 'g_in' in points. Use standardize_points(target_unit='mGal')."
        )

    # ensure z exists
    pts = points_gdf.copy()
    if "z" not in pts.columns:
        pts["z"] = np.nan

    z_before = pts["z"].astype(float).to_numpy()
    missing_before = ~np.isfinite(z_before)
    n_missing_before = int(missing_before.sum())

    # Clip DEM by AOI (bbox + outer_radius)
    aoi_pts = aoi_from_points_bbox(pts, buffer_m=float(outer_radius_m))

    with rasterio.open(dem_path) as src:
        dem_crs = src.crs.to_string() if src.crs else ""
    if not dem_crs:
        raise PreprocessError(f"DEM has no CRS: {dem_path}")

    aoi_dem = _reproject_shapely_geom(aoi_pts, pts.crs.to_string(), dem_crs)

    out_dem_clipped = Path(out_dem_clipped)
    out_points = Path(out_points)
    out_dem_clipped.parent.mkdir(parents=True, exist_ok=True)
    out_points.parent.mkdir(parents=True, exist_ok=True)

    if overwrite and out_dem_clipped.exists():
        out_dem_clipped.unlink()
    if overwrite and out_points.exists():
        out_points.unlink()

    clip_dem_to_geom(dem_path, aoi_dem, out_dem_clipped, overwrite=False)

    # Fill missing z from clipped DEM (only missing)
    pts_filled, _ = sample_dem_to_points(
        out_dem_clipped,
        pts,
        z_col="z",
        fill_only_missing=True,
        error_if_outside=False,
    )

    z_after = pts_filled["z"].astype(float).to_numpy()
    n_missing_after = int((~np.isfinite(z_after)).sum())
    n_filled_from_dem = int((missing_before & np.isfinite(z_after)).sum())

    # Build prisms from clipped DEM
    prisms, dens, dem_crs2 = dem_to_prisms(
        out_dem_clipped, density_kgm3=float(terrain_density_kgm3), reference_m=0.0
    )

    # Compute terrain effect at observation points (in DEM CRS)
    pts_dem = pts_filled.to_crs(dem_crs2)
    xs = np.asarray(pts_dem.geometry.x, dtype=float)
    ys = np.asarray(pts_dem.geometry.y, dtype=float)

    # Harmonica uses "upward" positive upward; g_z returns downward component in mGal (positive density -> positive anomaly)
    # so we pass upward = z + epsilon (meters). :contentReference[oaicite:2]{index=2}
    upward = np.asarray(pts_filled["z"], dtype=float) + float(station_epsilon_m)

    terrain_effect_mgal = hm.prism_gravity((xs, ys, upward), prisms, dens, field="g_z")

    slab_mgal = bouguer_slab_mgal(
        np.asarray(pts_filled["z"], dtype=float), float(bouguer_density_kgm3)
    )

    # Terrain correction to add to Bouguer anomaly
    tc_mgal = slab_mgal - terrain_effect_mgal

    cba_mgal = np.asarray(pts_filled["g_in"], dtype=float) + tc_mgal

    out = pts_filled.copy()
    out["slab_mgal"] = slab_mgal
    out["terrain_effect_mgal"] = terrain_effect_mgal
    out["tc_mgal"] = tc_mgal
    out["cba_mgal"] = cba_mgal
    out["bouguer_density_kgm3"] = float(bouguer_density_kgm3)
    out["terrain_density_kgm3"] = float(terrain_density_kgm3)
    out["tc_outer_radius_m"] = float(outer_radius_m)
    out["tc_station_epsilon_m"] = float(station_epsilon_m)
    out["tc_method"] = "dem_prism"

    write_gpkg(out, out_points, layer=layer, overwrite=False)

    return Step1Stats(
        n_points=int(len(out)),
        z_missing_before=n_missing_before,
        z_filled_from_dem=n_filled_from_dem,
        z_missing_after=n_missing_after,
        out_points=out_points,
        out_dem_clipped=out_dem_clipped,
        bouguer_density_kgm3=float(bouguer_density_kgm3),
        terrain_density_kgm3=float(terrain_density_kgm3),
        outer_radius_m=float(outer_radius_m),
    )
