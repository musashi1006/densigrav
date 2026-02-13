from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

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
    out_tc_grid: Optional[Path]
    out_cba_grid: Optional[Path]


def _require(mod: str, hint: str) -> None:
    try:
        __import__(mod)
    except Exception as e:
        raise PreprocessError(f"Missing dependency: {mod}. {hint}") from e


def _suggest_utm_epsg_from_lonlat(lon: float, lat: float) -> str:
    """
    Suggest UTM EPSG code based on lon/lat.
    North: EPSG:326xx, South: EPSG:327xx
    """
    zone = int(np.floor((lon + 180.0) / 6.0) + 1)
    zone = max(1, min(60, zone))
    if lat >= 0:
        return f"EPSG:326{zone:02d}"
    return f"EPSG:327{zone:02d}"


def _safe_unlink_sqlite(path: Path) -> None:
    # GeoPackage is SQLite: may have -wal/-shm sidecars
    for suffix in ["", "-wal", "-shm"]:
        p = Path(str(path) + suffix)
        if p.exists():
            try:
                p.unlink()
            except PermissionError as e:
                raise PreprocessError(
                    f"Cannot overwrite '{p}' (Permission denied).\n"
                    "Likely the file is open/locked by QGIS, sqlite3 or another process.\n"
                    "Close QGIS (remove the layer), then retry. "
                    "Alternatively change outputs.points_gpkg to a new file name."
                ) from e


def require_projected_crs(crs_str: str, points_gdf=None) -> None:
    """
    v0.1: require projected CRS (meters).
    If geographic, raise with a helpful suggestion (UTM) when possible.
    """
    _require("pyproj", "Install with: pip install -e '.[geo]'")
    from pyproj import CRS

    crs = CRS.from_user_input(crs_str)
    if not crs.is_geographic:
        return

    suggestion = None
    if points_gdf is not None and getattr(points_gdf, "geometry", None) is not None:
        try:
            # assume lon/lat when geographic
            lon = float(points_gdf.geometry.x.mean())
            lat = float(points_gdf.geometry.y.mean())
            suggestion = _suggest_utm_epsg_from_lonlat(lon, lat)
        except Exception:
            suggestion = None

    msg = (
        "CRS is geographic (degrees). v0.1 requires a projected CRS in meters (e.g., UTM).\n"
        f"Given: {crs_str}\n"
    )
    if suggestion:
        msg += f"Suggested projected CRS (UTM based on point centroid): {suggestion}\n"
        msg += "Tip: Reproject both points and DEM to the suggested CRS in QGIS, then set project.crs accordingly."
    else:
        msg += "Tip: Reproject both points and DEM to a projected CRS (meters), then set project.crs accordingly."

    raise PreprocessError(msg)


def bouguer_slab_mgal(height_m: np.ndarray, density_kgm3: float) -> np.ndarray:
    """
    Bouguer slab attraction in mGal:
      g = 2*pi*G*rho*h  [m/s^2] -> *1e5 [mGal]
    """
    g = 6.67430e-11
    return (2.0 * np.pi * g * float(density_kgm3) * height_m) * 1e5


def _read_dem_grid(dem_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, float]:
    """
    Return topo (ny,nx), xcenters(nx), ycenters(ny) increasing, crs_str, pixel_size_m
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
        dy = float(t.e)  # negative usually
        x0 = float(t.c)
        y0 = float(t.f)

        x = x0 + (np.arange(src.width) + 0.5) * dx
        y = y0 + (np.arange(src.height) + 0.5) * dy

    # ensure x increasing, y increasing
    if x.size >= 2 and x[1] < x[0]:
        x = x[::-1]
        topo = topo[:, ::-1]
        dx = -dx
    if y.size >= 2 and y[1] < y[0]:
        y = y[::-1]
        topo = topo[::-1, :]
        dy = -dy

    pixel = float(max(abs(dx), abs(dy)))
    return topo, x, y, crs, pixel


def dem_to_prisms(
    dem_path: Path,
    *,
    density_kgm3: float,
    reference_m: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Convert DEM grid to prisms array (n,6) and density array (n,).
    Only uses finite cells with top > reference.
    """
    topo, x, y, crs, pixel = _read_dem_grid(dem_path)

    top = np.where(np.isfinite(topo), topo, reference_m)
    top = np.maximum(top, reference_m)

    xx, yy = np.meshgrid(x, y)

    west = xx - pixel / 2.0
    east = xx + pixel / 2.0
    south = yy - pixel / 2.0
    north = yy + pixel / 2.0
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
    # Issue06 additions:
    regional_enabled: bool = False,
    regional_order: int = 2,
    regional_output: str = "residual",  # residual|regional|both
    grids_enabled: bool = False,
    grid_resolution_m: float = 250.0,
    out_tc_grid: Optional[Path] = None,
    out_cba_grid: Optional[Path] = None,
) -> Step1Stats:
    """
    Input points_gdf must have:
      - geometry (Point)
      - g_in (Bouguer anomaly) in mGal
      - z (optional; will be filled from DEM if missing)

    Output points gpkg adds:
      - slab_mgal, terrain_effect_mgal, tc_mgal, cba_mgal
      - (optional) regional_mgal, residual_mgal
    Output grids (optional):
      - tc_grid GeoTIFF
      - cba_grid GeoTIFF (field depends on regional_output)
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

    # v0.1: require projected CRS in meters (suggest UTM if geographic)
    require_projected_crs(points_gdf.crs.to_string(), points_gdf)

    if "g_in" not in points_gdf.columns:
        raise PreprocessError(
            "Missing required column 'g_in' (standardized Bouguer anomaly in mGal)."
        )

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
    if overwrite:
        _safe_unlink_sqlite(out_points)

    clip_dem_to_geom(dem_path, aoi_dem, out_dem_clipped, overwrite=False)

    # Fill missing z from clipped DEM (only missing)
    pts_filled, _ = sample_dem_to_points(out_dem_clipped, pts, z_col="z", fill_only_missing=True)
    z_after = pts_filled["z"].astype(float).to_numpy()
    n_missing_after = int((~np.isfinite(z_after)).sum())
    n_filled_from_dem = int((missing_before & np.isfinite(z_after)).sum())

    # Prisms from clipped DEM
    prisms, dens, dem_crs2 = dem_to_prisms(
        out_dem_clipped, density_kgm3=float(terrain_density_kgm3), reference_m=0.0
    )

    # Terrain effect at observation points (DEM CRS)
    pts_dem = pts_filled.to_crs(dem_crs2)
    xs = np.asarray(pts_dem.geometry.x, dtype=float)
    ys = np.asarray(pts_dem.geometry.y, dtype=float)
    upward = np.asarray(pts_filled["z"], dtype=float) + float(station_epsilon_m)

    terrain_effect_mgal = hm.prism_gravity((xs, ys, upward), prisms, dens, field="g_z")

    slab_mgal = bouguer_slab_mgal(
        np.asarray(pts_filled["z"], dtype=float), float(bouguer_density_kgm3)
    )
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

    # ---- Regional (Issue06)
    cba_for_grid = out["cba_mgal"].to_numpy(dtype=float)
    if regional_enabled:
        from densigrav.preprocess.regional import fit_polynomial_trend

        rr = fit_polynomial_trend(
            x=np.asarray(out.geometry.x, dtype=float),
            y=np.asarray(out.geometry.y, dtype=float),
            data=np.asarray(out["cba_mgal"], dtype=float),
            order=int(regional_order),
        )
        out["regional_mgal"] = rr.regional
        out["residual_mgal"] = rr.residual

        out_mode = (regional_output or "residual").strip().lower()
        if out_mode in ("regional",):
            cba_for_grid = out["regional_mgal"].to_numpy(dtype=float)
        else:
            # residual or both -> residual is the default "anomaly of interest"
            cba_for_grid = out["residual_mgal"].to_numpy(dtype=float)

    # write points
    write_gpkg(out, out_points, layer=layer, overwrite=False)

    # ---- Grids (Issue06)
    tc_grid_path = None
    cba_grid_path = None

    if grids_enabled:
        if out_tc_grid is None or out_cba_grid is None:
            raise PreprocessError("grids_enabled=True requires out_tc_grid and out_cba_grid paths.")

        from densigrav.preprocess.gridding import grid_from_points, write_geotiff

        x = np.asarray(out.geometry.x, dtype=float)
        y = np.asarray(out.geometry.y, dtype=float)
        crs_wkt = out.crs.to_string()

        # TC grid
        tc_res = grid_from_points(
            x=x,
            y=y,
            values=np.asarray(out["tc_mgal"], dtype=float),
            crs_wkt=crs_wkt,
            resolution_m=float(grid_resolution_m),
            region=None,
        )
        tc_grid_path = write_geotiff(
            out_path=Path(out_tc_grid),
            grid=tc_res.grid,
            transform=tc_res.transform,
            crs_wkt=tc_res.crs_wkt,
            overwrite=bool(overwrite),
        )

        # CBA (or residual/regional) grid
        cba_res = grid_from_points(
            x=x,
            y=y,
            values=np.asarray(cba_for_grid, dtype=float),
            crs_wkt=crs_wkt,
            resolution_m=float(grid_resolution_m),
            region=None,
        )
        cba_grid_path = write_geotiff(
            out_path=Path(out_cba_grid),
            grid=cba_res.grid,
            transform=cba_res.transform,
            crs_wkt=cba_res.crs_wkt,
            overwrite=bool(overwrite),
        )

    return Step1Stats(
        n_points=int(len(out)),
        z_missing_before=n_missing_before,
        z_filled_from_dem=n_filled_from_dem,
        z_missing_after=n_missing_after,
        out_points=out_points,
        out_dem_clipped=out_dem_clipped,
        out_tc_grid=tc_grid_path,
        out_cba_grid=cba_grid_path,
    )
